package es

import (
	"testing"

	"github.com/is8ac/tfutils"
	"github.com/is8ac/tfutils/tb"
	tf "github.com/tensorflow/tensorflow/tensorflow/go"
	"github.com/tensorflow/tensorflow/tensorflow/go/op"
)

func nilPerturb(s *op.Scope, input tf.Output, numOutputs tf.Output, generation tf.Output, globalSeed int64) func(*op.Scope, tf.Output) tf.Output {
	return func(subS *op.Scope, index tf.Output) (output tf.Output) {
		output = input
		return
	}
}

func nilLoss(s *op.Scope, actual, target tf.Output) (loss tf.Output) {
	loss = op.Const(s, float32(0))
	return
}

func TestNewSession(t *testing.T) {
	model := ModelDef{
		ParamDefs: []ParamDef{
			ParamDef{Name: "weights", ZeroVal: tfutils.Zero(tf.Float), Shape: tf.ScalarShape(), PerturbFunc: nilPerturb},
		},
		Model: func(s *op.Scope, params []tf.Output, input tf.Output) (output tf.Output, _ []tb.LogOP) {
			output = input
			return
		},
	}
	s := op.NewScope()
	inputs := op.Const(s.SubScope("inputs"), [][]float32{[]float32{1, 2}, []float32{3, 4}})
	targets := op.Const(s.SubScope("targets"), [][]float32{[]float32{0, 1}, []float32{1, 0}})
	esSess, err := NewSession(s.SubScope("main"), model, nilLoss, nilLoss, inputs, targets, inputs, targets, nil, 3, 42, "tb_logs", "test").Finalize()
	if err != nil {
		panic(err)
	}
	bestIndex, err := esSess.BestIndex()
	if err != nil {
		t.Fatal(err)
	}
	err = esSess.Perturb(bestIndex)
	if err != nil {
		t.Fatal(err)
	}
	_, err = esSess.Accuracy(false, false)
	if err != nil {
		t.Fatal(err)
	}
	_, err = esSess.Accuracy(true, true)
	if err != nil {
		t.Fatal(err)
	}
	err = esSess.Close()
	if err != nil {
		t.Fatal(err)
	}
}
