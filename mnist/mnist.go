package mnist

import (
	"compress/gzip"
	"fmt"
	"io"
	"net/http"
	"os"

	"github.com/is8ac/tfutils/quant"

	tf "github.com/tensorflow/tensorflow/tensorflow/go"
	"github.com/tensorflow/tensorflow/tensorflow/go/op"
)

const urlBase = "http://yann.lecun.com/exdb/mnist/"

// BasePath is the dir to which mnist data is looked for and saved
const BasePath = "mnist/"

func downloadData(name string) (err error) {
	if _, fsErr := os.Stat(BasePath + name); !os.IsNotExist(fsErr) {
		return
	}
	file, err := os.Create(BasePath + name)
	defer file.Close()
	if err != nil {
		return
	}
	url := urlBase + name + ".gz"
	fmt.Println("downloading", url)
	resp, err := http.Get(url)
	defer resp.Body.Close()
	if err != nil {
		return
	}
	gz, err := gzip.NewReader(resp.Body)
	if err != nil {
		return err
	}
	defer gz.Close()
	_, err = io.Copy(file, gz)
	if err != nil {
		return
	}
	return
}

// Download all the mnist data set
func Download() (err error) {
	names := []string{
		"train-images-idx3-ubyte",
		"train-labels-idx1-ubyte",
		"t10k-images-idx3-ubyte",
		"t10k-labels-idx1-ubyte",
	}
	err = os.MkdirAll(BasePath, os.ModePerm)
	if err != nil {
		return
	}
	for _, name := range names {
		err = downloadData(name)
		if err != nil {
			return
		}
	}
	return
}

// FlattenImages turns int8 image of shape [?, 28, 28] into float32 [?, 784]
func FlattenImages(s *op.Scope, intImages tf.Output) (flatenedFloats tf.Output) {
	flatenedFloats = op.Div(s,
		op.Cast(s,
			op.Reshape(s, intImages, op.Const(s.SubScope("shape"), []int64{-1, 28 * 28})),
			tf.Float,
		),
		op.Const(s.SubScope("255"), float32(255)),
	)
	return
}

// OneHotLabels converts int labels to oneHot encoded float arrays
func OneHotLabels(s *op.Scope, intLabels tf.Output) (oneHot tf.Output) {
	oneHot = op.OneHot(s,
		intLabels,
		op.Const(s.SubScope("depth"), int32(10)),
		op.Const(s.SubScope("1"), float32(1)),
		op.Const(s.SubScope("0"), float32(0)),
	)
	return
}

// LabelsTest returns an op to load the mnist test labels from a file as [10000] uint8
func LabelsTest(s *op.Scope) (labels tf.Output) {
	scope := s.SubScope("test_labels")
	labels = loadLabels(scope, "t10k-labels-idx1-ubyte", 10000)
	return
}

// LabelsTrain returns an op to load the mnist training labels from a file as [60000] uint8
func LabelsTrain(s *op.Scope) (labels tf.Output) {
	scope := s.SubScope("train_labels")
	labels = loadLabels(scope, "train-labels-idx1-ubyte", 60000)
	return
}

// ImagesTest returns an op to load the mnist training images from a file as [60000, 28, 28] uint8
func ImagesTest(s *op.Scope) (labels tf.Output) {
	labels = loadImages(s.SubScope("mnist_images_test"), "t10k-images-idx3-ubyte", 10000)
	return
}

// ImagesTrain returns an op to load the mnist training images from a file as [60000, 28, 28] uint8
func ImagesTrain(s *op.Scope) (labels tf.Output) {
	labels = loadImages(s.SubScope("mnist_images_train"), "train-images-idx3-ubyte", 60000)
	return
}

func loadLabels(s *op.Scope, name string, size int32) (labels tf.Output) {
	fileName := op.Const(s.SubScope("filename"), BasePath+name)
	fileBytes := op.ReadFile(s, fileName)
	ints := op.DecodeRaw(s, fileBytes, tf.Uint8, op.DecodeRawLittleEndian(true))
	trimHeader := op.Slice(s, ints,
		op.Const(s.SubScope("begin"), []int32{int32(8)}),
		op.Const(s.SubScope("size"), []int32{size}),
	)
	labels = op.Reshape(s, trimHeader, op.Const(s.SubScope("target_shape"), []int32{size}))
	return
}

func loadImages(s *op.Scope, name string, size int32) (labels tf.Output) {
	fileName := op.Const(s.SubScope("filename"), BasePath+name)
	fileBytes := op.ReadFile(s, fileName)
	ints := op.DecodeRaw(s, fileBytes, tf.Uint8, op.DecodeRawLittleEndian(true))
	trimHeader := op.Slice(s, ints,
		op.Const(s.SubScope("begin"), []int32{int32(16)}),
		op.Const(s.SubScope("size"), []int32{size * 28 * 28}),
	)
	labels = op.Reshape(s, trimHeader,
		op.Const(s.SubScope("target_shape"),
			[]int32{size, int32(28), int32(28)},
		))
	return
}

// Quantize01Floats quantizes float32 tensors between 0 and 1.
func Quantize01Floats(s *op.Scope, floats tf.Output) (output quant.Output) {
	zero := op.Const(s.SubScope("zero"), float32(0))
	one := op.Const(s.SubScope("one"), float32(1))
	output = quant.Wrap(op.QuantizeV2(s, floats, zero, one, tf.Quint8))
	return
}

// TrainingQueue returns a queue of label - image pairs.
// You must run the enqueue OP at least once before using queue output.
func TrainingQueue(s *op.Scope) (label, image tf.Output, enqueue *tf.Operation) {
	trainLabels := LabelsTrain(s)
	trainImages := ImagesTrain(s)
	dataType := []tf.DataType{tf.Uint8, tf.Uint8}
	//dataShapes := []tf.Shape{tf.ScalarShape(), tf.MakeShape(28, 28)}
	//queue := op.FIFOQueueV2(s, dataType, op.FIFOQueueV2Shapes(dataShapes))
	queue := op.RandomShuffleQueueV2(s, dataType)
	enqueue = op.QueueEnqueueManyV2(s, queue, []tf.Output{trainLabels, trainImages})
	components := op.QueueDequeueV2(s, queue, dataType)
	label = components[0]
	image = components[1]
	return
}

// NextBatch returns a data set of random minibatches of size n of pairs of labels and images.
// It is equivalent to mnist.train.next_batch(n) in the python mnist lib.
// Images are [784] floats.
// Deterministic if seed is non 0. If 0, random seed is used.
func NextBatch(s *op.Scope, n int64, seed int64) (batchLabels, batchImages tf.Output, init *tf.Operation) {
	seedOutput := op.Const(s, seed)
	outputTypes := []tf.DataType{tf.Float, tf.Float}
	outputShapes := []tf.Shape{tf.MakeShape(n, 10), tf.MakeShape(n, 28*28)}
	preBatchOutputShapes := []tf.Shape{tf.ScalarShape(), tf.MakeShape(28 * 28)}
	labels := OneHotLabels(s, LabelsTrain(s))
	images := FlattenImages(s, ImagesTrain(s))
	dataset := op.TensorSliceDataset(s, []tf.Output{labels, images}, preBatchOutputShapes)
	repeatDataset := op.RepeatDataset(s, dataset, op.Const(s.SubScope("count"), int64(-1)), outputTypes, preBatchOutputShapes)
	shuffleDataset := op.ShuffleDataset(s,
		repeatDataset,
		op.Const(s.SubScope("buffer_size"), int64(100000)),
		seedOutput,
		seedOutput,
		outputTypes,
		preBatchOutputShapes,
	)
	batchDataset := op.BatchDataset(s, shuffleDataset, op.Const(s.SubScope("batch_size"), n), outputTypes, outputShapes)
	iterator := op.Iterator(s, "", "", outputTypes, outputShapes)
	next := op.IteratorGetNext(s, iterator, outputTypes, outputShapes)
	init = op.MakeIterator(s, batchDataset, iterator)
	batchLabels = next[0]
	batchImages = next[1]
	return
}

// QuantizedNextBatch returns a data set of random minibatches of size n of pairs of labels and images.
// It is equivalent to mnist.train.next_batch(n) in the python mnist lib.
// Images are [784] quints.
// Deterministic if seed is non 0. If 0, random seed is used.
func QuantizedNextBatch(s *op.Scope, n int64, seed int64) (batchLabels, batchImages quant.Output, init *tf.Operation) {
	seedOutput := op.Const(s, seed)
	outputTypes := []tf.DataType{tf.Quint8, tf.Quint8}
	outputShapes := []tf.Shape{tf.MakeShape(n, 10), tf.MakeShape(n, 28*28)}
	preBatchOutputShapes := []tf.Shape{tf.ScalarShape(), tf.MakeShape(28 * 28)}
	labels := OneHotLabels(s, LabelsTrain(s))
	images := FlattenImages(s, ImagesTrain(s))

	qImages := Quantize01Floats(s.SubScope("images"), images)
	qLabels := Quantize01Floats(s.SubScope("labels"), labels)
	dataset := op.TensorSliceDataset(s, []tf.Output{qLabels.Output, qImages.Output}, preBatchOutputShapes)
	repeatDataset := op.RepeatDataset(s, dataset, op.Const(s.SubScope("count"), int64(-1)), outputTypes, preBatchOutputShapes)
	shuffleDataset := op.ShuffleDataset(s,
		repeatDataset,
		op.Const(s.SubScope("buffer_size"), int64(10000)),
		seedOutput,
		seedOutput,
		outputTypes,
		preBatchOutputShapes,
	)
	batchDataset := op.BatchDataset(s, shuffleDataset, op.Const(s.SubScope("batch_size"), n), outputTypes, outputShapes)
	iterator := op.Iterator(s, "", "", outputTypes, outputShapes)
	next := op.IteratorGetNext(s, iterator, outputTypes, outputShapes)
	init = op.MakeIterator(s, batchDataset, iterator)
	batchLabels = quant.Wrap(next[0], qLabels.Min, qLabels.Max)
	batchImages = quant.Wrap(next[1], qImages.Min, qImages.Max)
	return
}
