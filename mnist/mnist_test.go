package mnist

import (
	"fmt"
	"testing"

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
