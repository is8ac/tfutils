// Package models provides functions to construct simple models.
package models

import (
	"github.com/is8ac/tfutils"
	"github.com/is8ac/tfutils/descend"
	tf "github.com/tensorflow/tensorflow/tensorflow/go"
	"github.com/tensorflow/tensorflow/tensorflow/go/op"
)

// MakeSingleLayerNN create a modelDef for a single layer nn.
func MakeSingleLayerNN(inputs, targets tf.Output) (
	paramDefs []descend.ParamDef, // list of param tensors to have the state machine to create.
	lossFunc descend.LossFunc, // func for the state machine to evaluate parameters.
	makeFinalizeAccuracy func(*op.Scope, []tf.Output, tf.Output, tf.Output) func(*tf.Session) func() (float32, error), // func to make a func to make a func compute accuracy.
) {
	inputDims, err := inputs.Shape().ToSlice()
	if err != nil {
		panic(err)
	}
	targetDims, err := targets.Shape().ToSlice()
	if err != nil {
		panic(err)
	}
	if len(inputDims) != 2 {
		panic("input must be 2 dimensional, is shape " + inputs.Shape().String())
	}
	if len(targetDims) != 2 {
		panic("target must be 2 dimensional, is shape " + inputs.Shape().String())
	}
	model := func(s *op.Scope, params []tf.Output, inputs tf.Output) tf.Output {
		return op.Add(s, params[1], op.MatMul(s, inputs, params[0]))
	}
	paramDefs = []descend.ParamDef{
		descend.ParamDef{Name: "weights", Init: tfutils.Zero(tf.Float, tf.MakeShape(inputDims[1], targetDims[1]))},
		descend.ParamDef{Name: "biases", Init: tfutils.Zero(tf.Float, tf.MakeShape(targetDims[1]))},
	}
	lossFunc = func(s *op.Scope, params []tf.Output) (loss tf.Output) {
		softmax := op.Softmax(s, model(s, params, inputs))
		sqrDiffs := op.SquaredDifference(s, softmax, targets)
		sums := op.Sum(s, sqrDiffs, op.Const(s, int32(-1)))
		loss = op.Mean(s, sums, op.Const(s.SubScope("mean_reduce_dims"), []int32{0}))
		return
	}
	makeFinalizeAccuracy = func(s *op.Scope, params []tf.Output, testInputs, testTargets tf.Output) func(*tf.Session) func() (float32, error) {
		actual := model(s, params, testInputs)
		actualLabels := op.ArgMax(s, actual, op.Const(s.SubScope("argmax_dim"), int32(-1)), op.ArgMaxOutputType(tf.Int32))
		correct := op.Reshape(s, op.Equal(s, actualLabels, testTargets), op.Const(s.SubScope("all"), []int32{-1}))
		accuracy := op.Mean(s, op.Cast(s.SubScope("accuracy"), correct, tf.Float), op.Const(s.SubScope("mean_dim"), int32(0)))
		return func(sess *tf.Session) func() (float32, error) {
			return func() (acc float32, err error) {
				results, err := sess.Run(nil, []tf.Output{accuracy}, nil)
				if err != nil {
					return
				}
				acc = results[0].Value().(float32)
				return
			}
		}
	}
	return
}
