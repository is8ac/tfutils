package es

import (
	"fmt"
	"testing"

	"github.com/is8ac/tfutils/mnist"
	tf "github.com/tensorflow/tensorflow/tensorflow/go"
	"github.com/tensorflow/tensorflow/tensorflow/go/op"
)

func TestNewSession(t *testing.T) {
	const seed int64 = 42
	const numChildren int = 10
	const batchSize int64 = 1000
	err := mnist.Download()
	if err != nil {
		panic(err)
	}
	s := op.NewScope()

	labels, images, init := mnist.NextBatch(s.SubScope("next_batch"), batchSize, seed)
	initTestLabels, testLabels := varCache(s.SubScope("testLabels"), mnist.LabelsTest(s), "test_labels")
	initTestImages, testImages := varCache(s.SubScope("testImages"), mnist.FlattenImages(s.SubScope("flatten_images"), mnist.ImagesTest(s)), "test_images")
	initOPs := []*tf.Operation{
		init,
		initTestLabels,
		initTestImages,
	}
	esSess, err := NewSession(s.SubScope("main"), SingleLayerNN(28*28, 10), softmaxSqrDifLoss, PercentAccuracy, images, labels, testImages, testLabels, initOPs, numChildren, seed)
	if err != nil {
		panic(err)
	}
	//esSess.WriteTBgraph("tb_logs/graphs")
	for i := 0; i < 10; i++ {
		bestIndex := esSess.BestIndex()
		esSess.Perturb(bestIndex)
	}
	fmt.Println(esSess.Accuracy())
}
