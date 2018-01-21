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
	Model     func(s *op.Scope, params []tf.Output, x tf.Output, h0 tf.Output, c0 tf.Output) (output, h, c tf.Output, tbOps []tb.LogOP)
	InitHval  func(s *op.Scope) tf.Output
	Hsize     int64
}

// AddPost returns a new CellDef with the provided modelDef added a a post processing layer.
func (cd CellDef) AddPost(md es.ModelDef) (newcd CellDef) {
	for i := range md.ParamDefs {
		md.ParamDefs[i].Name = "post/" + md.ParamDefs[i].Name
	}
	newcd.ParamDefs = append(cd.ParamDefs, md.ParamDefs...)
	newcd.Model = func(s *op.Scope, params []tf.Output, x tf.Output, h0 tf.Output, c0 tf.Output) (output, h, c tf.Output, tbOPs []tb.LogOP) {
		if len(params) != len(cd.ParamDefs)+len(md.ParamDefs) {
			panic("wanted " + strconv.Itoa(len(params)) + " params, got " + strconv.Itoa(len(cd.ParamDefs)+len(md.ParamDefs)))
		}
		var rnnOutput tf.Output
		var modeltbOps []tb.LogOP
		rnnOutput, h, c, tbOPs = cd.Model(s.SubScope("rnn_cell"), params[0:len(cd.ParamDefs)], x, h0, c0)
		output, tbOPs = md.Model(s.SubScope("post_processing"), params[len(cd.ParamDefs):], rnnOutput)
		tbOPs = append(tbOPs, modeltbOps...)
		return
	}
	newcd.InitHval = cd.InitHval
	newcd.Hsize = cd.Hsize
	return
}

// RNN makes a simple RNNcell.
func RNN(dataType tf.DataType, inputSize, outputSize int64) (cell CellDef) {
	cell.ParamDefs = []es.ParamDef{
		es.ParamDef{Name: "weights", ZeroVal: tfutils.Zero(dataType), Shape: tf.MakeShape(inputSize+outputSize, outputSize), PerturbFunc: perturb.MakeSimple(0.003)},
		es.ParamDef{Name: "biases", ZeroVal: tfutils.Zero(dataType), Shape: tf.MakeShape(outputSize), PerturbFunc: perturb.MakeSimple(0.003)},
	}
	cell.Model = func(s *op.Scope, params []tf.Output, x tf.Output, h0 tf.Output, c0 tf.Output) (output, h, c tf.Output, tbOPs []tb.LogOP) {
		if len(params) != len(cell.ParamDefs) {
			panic("wanted " + strconv.Itoa(len(params)) + " params, got " + strconv.Itoa(len(cell.ParamDefs)))
		}
		input := op.Concat(s, op.Const(s, int32(1)), []tf.Output{x, h0})
		h = op.Tanh(s, op.Add(s, params[1], op.MatMul(s, input, params[0])))
		output = h
		return
	}
	cell.InitHval = tfutils.Zero(dataType)
	cell.Hsize = outputSize
	return
}

// for lstm
func slicePart(s *op.Scope, internal tf.Output, size, index int64) (output tf.Output) {
	return op.Slice(s,
		internal,
		op.Const(s.SubScope("begin"), []int64{0, size * index}),
		op.Const(s.SubScope("size"), []int64{-1, size}),
	)
}

// LSTM makes an LSTM cell.
func LSTM(dataType tf.DataType, inputSize, outputSize int64) (cell CellDef) {
	cell.ParamDefs = []es.ParamDef{
		es.ParamDef{Name: "weights", ZeroVal: tfutils.Zero(dataType), Shape: tf.MakeShape(inputSize+outputSize, outputSize*4), PerturbFunc: perturb.MakeSlice(0.003)},
		es.ParamDef{Name: "biases", ZeroVal: tfutils.Zero(dataType), Shape: tf.MakeShape(outputSize * 4), PerturbFunc: perturb.MakeSlice(0.003)},
	}
	cell.Model = func(s *op.Scope, params []tf.Output, x tf.Output, h0 tf.Output, c0 tf.Output) (output, h, c tf.Output, tbOPs []tb.LogOP) {
		input := op.Concat(s, op.Const(s, int32(1)), []tf.Output{x, h0})
		is := s.SubScope("internal")
		internal := op.Add(is, params[1], op.MatMul(is, input, params[0])) // `[batches, outputSize * 4]`
		fgs := s.SubScope("forget_gate")
		forgetGate := op.Sigmoid(fgs, slicePart(fgs, internal, outputSize, 0))
		forgetedC := op.Mul(fgs, forgetGate, c0)

		igs := s.SubScope("input_gate")
		inputGate := op.Sigmoid(igs, slicePart(igs, internal, outputSize, 1))

		uvs := s.SubScope("update_vals")
		updateCandidates := op.Tanh(uvs, slicePart(uvs, internal, outputSize, 2))
		updateVals := op.Mul(s, inputGate, updateCandidates)
		c = op.Add(s, forgetedC, updateVals)

		ogs := s.SubScope("output_gate")
		outputGate := op.Sigmoid(ogs, slicePart(ogs, internal, outputSize, 3))
		h = op.Mul(ogs, outputGate, op.Tanh(ogs, c))
		output = h
		return
	}
	cell.InitHval = tfutils.Zero(dataType)
	cell.Hsize = outputSize
	return
}

// Unroll converts an RNNcell into a normal modelDef.
// pre and post are inserted before and after the rnn cell.
// If you need no processing, use models.Nil for pre and/or post.
// Only the tensorboard summary operations from the last cell will be passed through.
func Unroll(n int64, cell CellDef) (md es.ModelDef) {
	md.ParamDefs = cell.ParamDefs
	md.Model = func(s *op.Scope, vars []tf.Output, inputs tf.Output) (outputs tf.Output, tbOPs []tb.LogOP) {
		inputList := op.Unpack(s, inputs, n, op.UnpackAxis(1))
		outputList := make([]tf.Output, n)
		inputShape := op.Shape(s, inputs, op.ShapeOutType(tf.Int32))
		if s.Err() != nil {
			panic(s.Err())
		}
		batchSize := op.Unpack(s.SubScope("batches"), inputShape, 3, op.UnpackAxis(0))[0]
		hShape := op.Pack(s.SubScope("multiples"), []tf.Output{batchSize, op.Const(s.SubScope("h_size"), int32(cell.Hsize))})
		h := op.Fill(s, hShape, cell.InitHval(s))
		c := h // c and h are the same shape, so let us use the same init value
		var output tf.Output
		for i, input := range inputList {
			output, h, c, tbOPs = cell.Model(s.SubScope("cell_"+strconv.Itoa(i)), vars, input, h, c)
			outputList[i] = output
		}
		outputs = op.Pack(s, outputList, op.PackAxis(1))
		return
	}
	return
}
