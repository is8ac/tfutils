package rnn

import (
	"strconv"

	"github.com/is8ac/tfutils"
	"github.com/is8ac/tfutils/es"
	"github.com/is8ac/tfutils/es/perturb"
	"github.com/is8ac/tfutils/tb"
	tf "github.com/tensorflow/tensorflow/tensorflow/go"
	"github.com/tensorflow/tensorflow/tensorflow/go/op"
)

// CellDef is like es.ModelDef, but holds recurrent cells.
type CellDef struct {
	ParamDefs []es.ParamDef
	Model     func(s *op.Scope, params []tf.Output, x tf.Output, h0 tf.Output, c0 []tf.Output) (h tf.Output, c []tf.Output, tbOps []tb.LogOP)
	InitHval  func(s *op.Scope) tf.Output
	Hsize     int64
}

// SimpleRNN makes a is a simple RNNcell.
func SimpleRNN(dataType tf.DataType, inputSize, outputSize int64) (cell CellDef) {
	cell.ParamDefs = []es.ParamDef{
		es.ParamDef{Name: "weights", ZeroVal: tfutils.Zero(dataType), Shape: tf.MakeShape(inputSize+outputSize, outputSize), PerturbFunc: perturb.MakeSlice(0.003)},
		es.ParamDef{Name: "biases", ZeroVal: tfutils.Zero(dataType), Shape: tf.MakeShape(outputSize), PerturbFunc: perturb.MakeSlice(0.003)},
	}
	cell.Model = func(s *op.Scope, params []tf.Output, x tf.Output, h0 tf.Output, c0 []tf.Output) (h tf.Output, c []tf.Output, tbOPs []tb.LogOP) {
		input := op.Concat(s, op.Const(s, int32(1)), []tf.Output{x, h0})
		h = op.Tanh(s, op.Add(s, params[1], op.MatMul(s, input, params[0])))
		return
	}
	cell.InitHval = tfutils.Zero(dataType)
	cell.Hsize = outputSize
	return
}

// Unroll converts an RNNcell into a normal modelDef.
// pre and post are inserted before and after the rnn cell.
// If you need no processing, use models.Nil for pre and/or post.
func Unroll(n int64, cell CellDef, pre, post es.ModelDef) (md es.ModelDef) {
	md.ParamDefs = cell.ParamDefs
	md.Model = func(s *op.Scope, vars []tf.Output, inputs tf.Output) (outputs tf.Output, tbOPs []tb.LogOP) {
		// input.Shape() == `[4, 3, 5]`
		inputList := op.Unpack(s, inputs, n, op.UnpackAxis(1))
		outputList := make([]tf.Output, n)
		inputShape := op.Shape(s, inputs, op.ShapeOutType(tf.Int32))
		inputIndexes := op.Unpack(s.SubScope("batches"), inputShape, 3, op.UnpackAxis(0))
		hShape := op.Pack(s.SubScope("multiples"), []tf.Output{inputIndexes[0], op.Const(s.SubScope("h_size"), int32(cell.Hsize))})
		h := op.Fill(s, hShape, cell.InitHval(s))
		c := []tf.Output{}
		for i, input := range inputList {
			h, _, _ = cell.Model(s.SubScope("cell_"+strconv.Itoa(i)), vars, input, h, c)
			outputList[i] = h
		}
		outputs = op.Pack(s, outputList, op.PackAxis(1))
		return
	}
	return
}
