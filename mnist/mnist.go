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

// FlattenImages turns a tensor of shape [?, 28, 28] into a tensor of shape [?, 784], and same shape
func FlattenImages(s *op.Scope, intImages tf.Output) (flattened tf.Output) {
	flattened = op.Reshape(s, intImages, op.Const(s.SubScope("shape"), []int64{-1, 28 * 28}))
	return
}

func makeConst(s *op.Scope, DstT tf.DataType, value float32) tf.Output {
	return op.Cast(s, op.Const(s, value), DstT)
}

// InitCastImages turns int8 from 0-255, to dType type from 0-1 of same shape.
func InitCastImages(DstT tf.DataType) func(*op.Scope, tf.Output) tf.Output {
	return func(s *op.Scope, intImages tf.Output) (floats tf.Output) {
		floats = op.Div(s,
			op.Cast(s, intImages, DstT),
			makeConst(s.SubScope("255"), DstT, float32(255)),
		)
		return
	}
}

// InitOneHotLabels converts int labels to oneHot encoded float arrays
func InitOneHotLabels(DstT tf.DataType) func(s *op.Scope, intLabels tf.Output) tf.Output {
	return func(s *op.Scope, intLabels tf.Output) (oneHot tf.Output) {
		one := makeConst(s.SubScope("one"), DstT, 1)
		zero := makeConst(s.SubScope("zero"), DstT, 0)
		oneHot = op.OneHot(s,
			intLabels,
			op.Const(s.SubScope("depth"), int32(10)),
			one,
			zero,
		)
		return
	}
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
// imagesTransform transforms images. If nil, float32 28x28 are returned.
// labelsTransform transforms labels. If nil, onehot floats are returned.
// Deterministic if seed is non 0. If 0, random seed is used.
func NextBatch(s *op.Scope, imagesTransform, labelsTransform func(*op.Scope, tf.Output) tf.Output, n int64, seed int64) (batchImages, batchLabels tf.Output, init *tf.Operation) {
	if imagesTransform == nil {
		imagesTransform = InitCastImages(tf.Float)
	}
	if labelsTransform == nil {
		labelsTransform = InitOneHotLabels(tf.Float)
	}
	images := imagesTransform(s, ImagesTrain(s))
	labels := labelsTransform(s, LabelsTrain(s))
	labelsShape, err := labels.Shape().ToSlice()
	if err != nil {
		panic(err)
	}
	imagesShape, err := images.Shape().ToSlice()
	if err != nil {
		panic(err)
	}
	imagesShape[0] = n
	labelsShape[0] = n
	outputTypes := []tf.DataType{images.DataType(), labels.DataType()}
	outputShapes := []tf.Shape{tf.MakeShape(imagesShape...), tf.MakeShape(labelsShape...)}
	preBatchOutputShapes := []tf.Shape{tf.MakeShape(imagesShape[1:]...), tf.MakeShape(labelsShape[1:]...)}
	seedOutput := op.Const(s, seed)
	dataset := op.TensorSliceDataset(s, []tf.Output{images, labels}, preBatchOutputShapes)
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
	batchImages = next[0]
	batchLabels = next[1]
	return
}
