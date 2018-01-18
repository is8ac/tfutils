package models

import (
	"os"
	"testing"

	"github.com/is8ac/tfutils"
	"github.com/is8ac/tfutils/es"
	"github.com/is8ac/tfutils/es/accuracy"
	"github.com/is8ac/tfutils/es/loss"
	"github.com/is8ac/tfutils/mnist"
	tf "github.com/tensorflow/tensorflow/tensorflow/go"
	"github.com/tensorflow/tensorflow/tensorflow/go/op"
)

func testModelMNIST(modelDef es.ModelDef, dType tf.DataType) (err error) {
	s := op.NewScope()

	images, labels, init := mnist.NextBatch(s.SubScope("next_batch"),
		func(s *op.Scope, input tf.Output) tf.Output {
			return mnist.InitCastImages(dType)(s, mnist.FlattenImages(s, input))
		},
		mnist.InitOneHotLabels(dType),
		30,
		42,
	)
	initTestLabels, testLabels := tfutils.VarCache(s.SubScope("testLabels"), mnist.LabelsTest(s), "test_labels")
	initTestImages, testImages := tfutils.VarCache(s.SubScope("testImages"), mnist.InitCastImages(dType)(s, mnist.FlattenImages(s.SubScope("flatten_images"), mnist.ImagesTest(s))), "test_images")
	initOPs := []*tf.Operation{
		init,
		initTestLabels,
		initTestImages,
	}
	esSess, err := es.NewSession(s.SubScope("main"), modelDef, loss.SoftmaxSqrDif, accuracy.Percent, images, labels, testImages, testLabels, initOPs, 5, 52, "tb_logs", "")
	if err != nil {
		panic(err)
	}
	var bestIndex int32
	for i := 0; i < 100; i++ {
		bestIndex, err = esSess.BestIndex()
		if err != nil {
			return
		}
		err = esSess.Perturb(bestIndex)
		if err != nil {
			return
		}
	}
	return
}

func TestTwoLayerNN(t *testing.T) {
	os.Setenv("TF_CPP_MIN_LOG_LEVEL", "2")
	err := mnist.Download()
	if err != nil {
		t.Fatal(err)
	}
	dTypes := []tf.DataType{tf.Half, tf.Float, tf.Double}
	for _, dType := range dTypes {
		if err := testModelMNIST(TwoLayerNN(dType, 28*28, 5, 10), dType); err != nil {
			t.Fatal(err)
		}
	}
}
