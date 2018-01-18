// Package models provides functions to construct simple models.
package models

import (
	"github.com/is8ac/tfutils"
	"github.com/is8ac/tfutils/es"
	"github.com/is8ac/tfutils/es/perturb"
	"github.com/is8ac/tfutils/tb"
	tf "github.com/tensorflow/tensorflow/tensorflow/go"
	"github.com/tensorflow/tensorflow/tensorflow/go/op"
)

// SelfContainedQsingleLayerNN create a modelDef for a single layer NN using 8bit quantised.
// input must be float between 0 and 1
// returns floats
func SelfContainedQsingleLayerNN(inputSize, outputSize int64, perturbStdev, perturbRange float32) (model es.ModelDef) {
	model.ParamDefs = []es.ParamDef{
		es.ParamDef{Name: "weights", ZeroVal: tfutils.Zero(tf.Quint8), Shape: tf.MakeShape(inputSize, outputSize), PerturbFunc: perturb.MakeQsimple(perturbStdev, perturbRange)},
		es.ParamDef{Name: "biases", ZeroVal: tfutils.Zero(tf.Quint8), Shape: tf.MakeShape(outputSize), PerturbFunc: perturb.MakeQsimple(perturbStdev, perturbRange)},
	}

	model.Model = func(s *op.Scope, vars []tf.Output, inputs tf.Output) (output tf.Output, tbOPs []tb.LogOP) {
		zero := op.Const(s.SubScope("zero"), float32(0)) // min range for input.
		one := op.Const(s.SubScope("one"), float32(1))   // max range for the input.
		perturbMin := op.Const(s.SubScope("perturb_min"), -perturbRange)
		perturbMax := op.Const(s.SubScope("perturb_max"), perturbRange)
		qInputs, _, _ := op.QuantizeV2(s, inputs, zero, one, tf.Quint8)                                                    // the input is gray scale for from 0 to 1.
		matmul32, matmul32Min, matmul32Max := op.QuantizedMatMul(s, qInputs, vars[0], zero, one, perturbMin, perturbMax)   // matmul inputs with weights
		matmul8, matmul8Min, matmul8Max := op.QuantizeDownAndShrinkRange(s, matmul32, matmul32Min, matmul32Max, tf.Quint8) // convert back down to quint8
		add32, add32Min, add32Max := op.QuantizedAdd(s, matmul8, vars[1], matmul8Min, matmul8Max, perturbMin, perturbMax)
		output = op.Dequantize(s, add32, add32Min, add32Max)
		tbOPs = []tb.LogOP{
			tb.MakeHistLogOP(op.Bitcast(s.SubScope("weights"), vars[0], tf.Uint8), "weights"),
			tb.MakeHistLogOP(op.Bitcast(s.SubScope("biases"), vars[1], tf.Uint8), "biases"),
		}
		return
	}
	return
}

// SingleLayerNN create a modelDef for a single layer nn. DstT must by a one of half, float, or double
func SingleLayerNN(DstT tf.DataType, inputSize, outputSize int64) (model es.ModelDef) {
	model.ParamDefs = []es.ParamDef{
		es.ParamDef{Name: "weights", ZeroVal: tfutils.Zero(DstT), Shape: tf.MakeShape(inputSize, outputSize), PerturbFunc: perturb.MakeSlice(0.003)},
		es.ParamDef{Name: "biases", ZeroVal: tfutils.Zero(DstT), Shape: tf.MakeShape(outputSize), PerturbFunc: perturb.MakeSlice(0.003)},
	}

	model.Model = func(s *op.Scope, vars []tf.Output, input tf.Output) (output tf.Output, tbOPs []tb.LogOP) {
		matmuled := op.MatMul(s, input, vars[0])
		output = op.Add(s, vars[1], matmuled)
		tbOPs = []tb.LogOP{
			tb.MakeHistLogOP(output, "output"),
			tb.MakeHistLogOP(vars[0], "weights"),
			tb.MakeHistLogOP(vars[1], "biases"),
			tb.MakeHistLogOP(matmuled, "matmuled"),
			tb.MakeHistLogOP(input, "input"),
			tb.MakePlusMinusOneImageLogOP(vars[0], "weights_image"),
		}
		return
	}
	return
}

// TwoLayerNN is a two layer fully interconnected neural net.
func TwoLayerNN(DstT tf.DataType, inputSize, hiddenSize, outputSize int64) (model es.ModelDef) {
	model.ParamDefs = []es.ParamDef{
		es.ParamDef{Name: "l1weights", ZeroVal: tfutils.Zero(DstT), Shape: tf.MakeShape(inputSize, hiddenSize), PerturbFunc: perturb.MakeSlice(0.003)},
		es.ParamDef{Name: "l1biases", ZeroVal: tfutils.Zero(DstT), Shape: tf.MakeShape(hiddenSize), PerturbFunc: perturb.MakeSlice(0.003)},
		es.ParamDef{Name: "l2weights", ZeroVal: tfutils.Zero(DstT), Shape: tf.MakeShape(hiddenSize, outputSize), PerturbFunc: perturb.MakeSlice(0.003)},
		es.ParamDef{Name: "l2biases", ZeroVal: tfutils.Zero(DstT), Shape: tf.MakeShape(outputSize), PerturbFunc: perturb.MakeSlice(0.003)},
	}

	model.Model = func(s *op.Scope, vars []tf.Output, images tf.Output) (output tf.Output, tbOPs []tb.LogOP) {
		l1s := s.SubScope("l1")
		l1 := op.Add(l1s, vars[1], op.MatMul(l1s, images, vars[0]))
		nonlin := op.Tanh(l1s, l1)
		l2s := s.SubScope("l2")
		output = op.Add(l2s, vars[3], op.MatMul(l2s, nonlin, vars[2]))
		tbOPs = []tb.LogOP{
			tb.MakeHistLogOP(output, "l2_output"),
			tb.MakeHistLogOP(l1, "l1_output"),
			tb.MakeHistLogOP(nonlin, "nonlin"),
			tb.MakeHistLogOP(vars[0], "l1_weights"),
			tb.MakeHistLogOP(vars[1], "l1_biases"),
			tb.MakeHistLogOP(vars[2], "l2_weights"),
			tb.MakeHistLogOP(vars[3], "l2_biases"),
		}
		return
	}
	return
}

// DummyModel is dummy
func DummyModel(outputSize int64) (model es.ModelDef) {
	model.ParamDefs = []es.ParamDef{
		es.ParamDef{Name: "biases", ZeroVal: tfutils.Zero(tf.Float), Shape: tf.MakeShape(outputSize), PerturbFunc: perturb.Nil},
	}
	model.Model = func(s *op.Scope, vars []tf.Output, images tf.Output) (output tf.Output, tbOPs []tb.LogOP) {
		output = op.Tile(s, op.ExpandDims(s, vars[0], op.Const(s.SubScope("dims"), int32(0))), op.Const(s, []int32{10000, 1}))
		return
	}
	return
}
