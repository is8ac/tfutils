package mnist

import (
	"compress/gzip"
	"fmt"
	"io"
	"net/http"
	"os"

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

// LabelsTest returns an op to load the mnist test labels from a file as [10000] uint8
func LabelsTest(s *op.Scope) (labels tf.Output) {
	labels = loadLabels(s.SubScope("mnist_labels_test"), "t10k-labels-idx1-ubyte", 10000)
	return
}

// LabelsTrain returns an op to load the mnist training labels from a file as [60000] uint8
func LabelsTrain(s *op.Scope) (labels tf.Output) {
	labels = loadLabels(s.SubScope("mnist_labels_train"), "train-labels-idx1-ubyte", 60000)
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
