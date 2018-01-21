package rnn

import (
	"testing"

	tf "github.com/tensorflow/tensorflow/tensorflow/go"
	"github.com/tensorflow/tensorflow/tensorflow/go/op"

	"github.com/is8ac/tfutils/es"
	"github.com/is8ac/tfutils/es/accuracy"
	"github.com/is8ac/tfutils/es/loss"
	"github.com/is8ac/tfutils/es/models"
)

func TestSimpleRNN(t *testing.T) {
	md := RNN(tf.Float, 3, 2)
	s := op.NewScope()
	params := make([]tf.Output, len(md.ParamDefs))
	for i, vd := range md.ParamDefs {
		varScope := s.SubScope(vd.Name)
		zero := vd.ZeroVal(varScope.SubScope("zero"))
		sliceShape, err := vd.Shape.ToSlice()
		if err != nil {
			panic(err)
		}
		params[i] = op.Fill(varScope, op.Cast(varScope, op.Const(varScope, sliceShape), tf.Int32), zero)
	}
	x := op.Const(s.SubScope("x"), [][]float32{[]float32{1, 2, 3}, []float32{1, 2, 3}, []float32{1, 2, 3}, []float32{1, 2, 3}})
	h0 := op.Const(s.SubScope("h0"), [][]float32{[]float32{0, 0}, []float32{0, 0}, []float32{0, 0}, []float32{0, 0}})
	_, h, _, _ := md.Model(s.SubScope("model"), params, x, h0, tf.Output{})
	graph, err := s.Finalize()
	if err != nil {
		t.Fatal(err)
	}
	sess, err := tf.NewSession(graph, nil)
	if err != nil {
		t.Fatal(err)
	}
	results, err := sess.Run(nil, []tf.Output{h}, nil)
	if err != nil {
		t.Fatal(err)
	}
	shape := results[0].Shape()
	if shape[0] != 4 {
		t.Fatal("wrong num batches")
	}
	if shape[1] != 2 {
		t.Fatal("wrong output size")
	}
}

func testUnroll(t *testing.T, cell CellDef) {
	md := Unroll(3, cell)
	s := op.NewScope()
	// `[seqLen, batchSize, inputSize]`
	inputs := op.Const(s.SubScope("inputs"), [][][]float32{
		[][]float32{[]float32{1, 2, 3, 4, 5}, []float32{3, 4, 3, 4, 5}, []float32{3, 4, 3, 4, 5}},
		[][]float32{[]float32{1, 2, 3, 4, 5}, []float32{3, 4, 3, 4, 5}, []float32{3, 4, 3, 4, 5}},
		[][]float32{[]float32{1, 2, 3, 4, 5}, []float32{3, 4, 3, 4, 5}, []float32{3, 4, 3, 4, 5}},
		[][]float32{[]float32{1, 2, 3, 4, 5}, []float32{3, 4, 3, 4, 5}, []float32{3, 4, 3, 4, 5}},
	})
	targets := op.Const(s.SubScope("targets"), [][][]float32{
		[][]float32{[]float32{1, 2}, []float32{3, 4}, []float32{3, 4}},
		[][]float32{[]float32{1, 2}, []float32{3, 4}, []float32{3, 4}},
		[][]float32{[]float32{1, 2}, []float32{3, 4}, []float32{3, 4}},
		[][]float32{[]float32{1, 2}, []float32{3, 4}, []float32{3, 4}},
	})
	testTargets := op.Const(s.SubScope("targets"), [][]int32{[]int32{1, 0, 1}, []int32{1, 0, 1}, []int32{1, 0, 1}, []int32{1, 0, 1}})
	esSess, err := es.NewSession(s.SubScope("main"), md, loss.SoftmaxSqrDif, accuracy.Percent, inputs, targets, inputs, testTargets, nil, 3, 42, "tb_logs", "").Finalize()
	if err != nil {
		t.Fatal(err)
	}
	bestIndex, err := esSess.BestIndex()
	err = esSess.Perturb(bestIndex)
	if err != nil {
		t.Fatal(err)
	}
}

func TestRNN(t *testing.T) {
	testUnroll(t, RNN(tf.Float, 5, 2))
}

func TestAddPost(t *testing.T) {
	testUnroll(t, RNN(tf.Float, 5, 3).AddPost(models.SingleLayerNN(tf.Float, 3, 2)))
}

func TestLSTM(t *testing.T) {
	testUnroll(t, LSTM(tf.Float, 5, 2))
}
