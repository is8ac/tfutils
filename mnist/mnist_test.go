package mnist

import (
	"fmt"
	"testing"

	"github.com/is8ac/tfutils/quant"

	tf "github.com/tensorflow/tensorflow/tensorflow/go"
	"github.com/tensorflow/tensorflow/tensorflow/go/op"
)

func TestDownloadData(t *testing.T) {
	err := Download()
	if err != nil {
		t.Fatal(err)
	}
}

func TestLabelsTest(t *testing.T) {
	s := op.NewScope()
	testLabels := LabelsTest(s)
	graph, err := s.Finalize()
	if err != nil {
		t.Fatal(err)
	}
	sess, err := tf.NewSession(graph, nil)
	if err != nil {
		t.Fatal(err)
	}
	results, err := sess.Run(nil, []tf.Output{testLabels}, nil)
	if err != nil {
		t.Fatal(err)
	}
	shape := results[0].Shape()
	if len(shape) != 1 {
		t.Fatal("wrong dims")
	}
	if shape[0] != 10000 {
		t.Fatal("wrong size")
	}
}

func TestLabelsTrain(t *testing.T) {
	s := op.NewScope()
	testLabels := LabelsTrain(s)
	graph, err := s.Finalize()
	if err != nil {
		t.Fatal(err)
	}
	sess, err := tf.NewSession(graph, nil)
	if err != nil {
		t.Fatal(err)
	}
	results, err := sess.Run(nil, []tf.Output{testLabels}, nil)
	if err != nil {
		t.Fatal(err)
	}
	shape := results[0].Shape()
	if len(shape) != 1 {
		t.Fatal("wrong dims")
	}
	if shape[0] != 60000 {
		fmt.Println(shape)
		t.Fatal("wrong size")
	}
}

func TestImagesTrain(t *testing.T) {
	s := op.NewScope()
	testLabels := ImagesTrain(s)
	graph, err := s.Finalize()
	if err != nil {
		t.Fatal(err)
	}
	sess, err := tf.NewSession(graph, nil)
	if err != nil {
		t.Fatal(err)
	}
	results, err := sess.Run(nil, []tf.Output{testLabels}, nil)
	if err != nil {
		t.Fatal(err)
	}
	shape := results[0].Shape()
	if len(shape) != 3 {
		fmt.Println(shape)
		t.Fatal("wrong dims")
	}
	if shape[0] != 60000 {
		fmt.Println(shape)
		t.Fatal("wrong size")
	}
}

func TestImagesTest(t *testing.T) {
	s := op.NewScope()
	testLabels := ImagesTest(s)
	graph, err := s.Finalize()
	if err != nil {
		t.Fatal(err)
	}
	sess, err := tf.NewSession(graph, nil)
	if err != nil {
		t.Fatal(err)
	}
	results, err := sess.Run(nil, []tf.Output{testLabels}, nil)
	if err != nil {
		t.Fatal(err)
	}
	shape := results[0].Shape()
	if len(shape) != 3 {
		fmt.Println(shape)
		t.Fatal("wrong dims")
	}
	if shape[0] != 10000 {
		fmt.Println(shape)
		t.Fatal("wrong size")
	}
}

func TestTrainQueue(t *testing.T) {
	s := op.NewScope()
	labels, images, enqueue := TrainingQueue(s)
	graph, err := s.Finalize()
	if err != nil {
		t.Fatal(err)
	}
	sess, err := tf.NewSession(graph, nil)
	if err != nil {
		t.Fatal(err)
	}
	_, err = sess.Run(nil, nil, []*tf.Operation{enqueue})
	if err != nil {
		t.Fatal(err)
	}
	results, err := sess.Run(nil, []tf.Output{labels, images}, nil)
	if err != nil {
		t.Fatal(err)
	}
	labelsShape := results[0].Shape()
	imagesShape := results[1].Shape()
	if len(labelsShape) != 0 {
		t.Fatal("wrong labels dim:", labelsShape)
	}
	if imagesShape[0] != 28 {
		t.Fatal("wrong images batch size")
	}
}

func TestNextBatch(t *testing.T) {
	s := op.NewScope()
	labels, images, init := NextBatch(s, 5, 1)
	graph, err := s.Finalize()
	if err != nil {
		t.Fatal(err)
	}
	sess, err := tf.NewSession(graph, nil)
	if err != nil {
		t.Fatal(err)
	}
	_, err = sess.Run(nil, nil, []*tf.Operation{init})
	if err != nil {
		t.Fatal(err)
	}
	results, err := sess.Run(nil, []tf.Output{labels, images}, nil)
	if err != nil {
		t.Fatal(err)
	}
	labelsShape := results[0].Shape()
	imagesShape := results[1].Shape()
	if len(labelsShape) != 2 {
		t.Fatal("wrong labels dim:", labelsShape)
	}
	if len(imagesShape) != 2 {
		t.Fatal("wrong images dim:", imagesShape)
	}
	if !(imagesShape[0] == 5 && labelsShape[0] == 5) {
		t.Fatal("wrong images batch size")
	}
}

func TestQuantizedNextBatch(t *testing.T) {
	s := op.NewScope()
	qLabels, qImages, init := QuantizedNextBatch(s, 5, 1)
	labelsBatch := op.Dequantize(quant.UnwrapS(s.SubScope("labels"), qLabels))
	imagesBatch := op.Dequantize(quant.UnwrapS(s.SubScope("images"), qImages))
	graph, err := s.Finalize()
	if err != nil {
		t.Fatal(err)
	}
	sess, err := tf.NewSession(graph, nil)
	if err != nil {
		t.Fatal(err)
	}
	_, err = sess.Run(nil, nil, []*tf.Operation{init})
	if err != nil {
		t.Fatal(err)
	}
	results, err := sess.Run(nil, []tf.Output{labelsBatch, imagesBatch}, nil)
	if err != nil {
		t.Fatal(err)
	}
	labelsShape := results[0].Shape()
	imagesShape := results[1].Shape()
	if len(labelsShape) != 2 {
		t.Fatal("wrong labels dim:", labelsShape)
	}
	if len(imagesShape) != 2 {
		t.Fatal("wrong images dim:", imagesShape)
	}
	if !(imagesShape[0] == 5 && labelsShape[0] == 5) {
		t.Fatal("wrong images batch size")
	}
}
