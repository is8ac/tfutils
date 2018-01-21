// Package perturb provides tools to construct various perturb funcs for es.
// All should be deterministic provided that globalSeed is non zero.
package perturb

import (
	"github.com/is8ac/tfutils/es"
	tf "github.com/tensorflow/tensorflow/tensorflow/go"
	"github.com/tensorflow/tensorflow/tensorflow/go/op"
)

// MakeQsimple returns a PerturbFunc which takes a quantaized tensor which must have range set my min and max.
func MakeQsimple(stdevVal, outputRange float32) es.PerturbFunc {
	return func(s *op.Scope, input tf.Output, numOutputs tf.Output, generation tf.Output, globalSeed int64) func(*op.Scope, tf.Output) tf.Output {
		max := op.Const(s.SubScope("max"), outputRange)
		min := op.Const(s.SubScope("min"), -outputRange)
		posStdev2 := op.Const(s.SubScope("stdev_max"), stdevVal*2)
		negStdev2 := op.Const(s.SubScope("stdev_min"), stdevVal*-2)
		nMax := op.Const(s.SubScope("nmax"), float32(2))
		nMin := op.Const(s.SubScope("nmin"), float32(-2))
		inputShape := op.Shape(s.SubScope("input"), input, op.ShapeOutType(tf.Int32))
		return func(subS *op.Scope, index tf.Output) (output tf.Output) {
			seed := op.Pack(subS, []tf.Output{
				generation,
				op.Cast(subS, index, tf.Int64),
			})
			noise := op.StatelessTruncatedNormal(subS, inputShape, seed, op.StatelessTruncatedNormalDtype(tf.Float))
			qNoise, _, _ := op.QuantizeV2(subS, noise, nMin, nMax, tf.Quint8)
			qp32, qpMin, qpMax := op.QuantizedAdd(subS, input, qNoise, min, max, negStdev2, posStdev2)
			output, _, _ = op.Requantize(subS, qp32, qpMin, qpMax, min, max, tf.Quint8)
			return
		}
	}
}

// MakeSlice returns a PerturbFunc which uses a slice of a single tensor. Better performance.
func MakeSlice(stdevVal float32) es.PerturbFunc {
	return func(s *op.Scope, input tf.Output, numOutputs tf.Output, generation tf.Output, globalSeed int64) func(*op.Scope, tf.Output) tf.Output {
		inputShape := op.Shape(s.SubScope("input"), input, op.ShapeOutType(tf.Int32))
		noiseShape := op.Add(s.SubScope("shape"), inputShape, numOutputs)
		seed := op.Pack(s, []tf.Output{generation, op.Const(s.SubScope("seed"), globalSeed)})
		oneNoise := tf.Output{}
		if input.DataType() == tf.Bfloat16 {
			oneNoise = op.Cast(s.SubScope("fix_bfloat16"), op.StatelessRandomNormal(s, noiseShape, seed, op.StatelessRandomNormalDtype(tf.Float)), tf.Bfloat16)
		} else {
			oneNoise = op.StatelessRandomNormal(s, noiseShape, seed, op.StatelessRandomNormalDtype(input.DataType()))
		}
		stdev := op.Cast(s, op.Const(s.SubScope("stdev"), stdevVal), input.DataType())
		noise := op.Mul(s, oneNoise, stdev)
		return func(subS *op.Scope, index tf.Output) (output tf.Output) {
			begin := op.Fill(subS.SubScope("size"), op.Shape(subS.SubScope("dims"), inputShape, op.ShapeOutType(tf.Int32)), index)
			noiseSlice := op.Slice(subS, noise, begin, inputShape)
			output = op.Add(subS, input, noiseSlice)
			return
		}
	}
}

// MakeSimple returns a PerturbFunc which uses generation and index as seeds.
func MakeSimple(stdevVal float32) es.PerturbFunc {
	return func(s *op.Scope, input tf.Output, numOutputs tf.Output, generation tf.Output, globalSeed int64) func(*op.Scope, tf.Output) tf.Output {
		inputShape := op.Shape(s.SubScope("input"), input, op.ShapeOutType(tf.Int32))
		return func(subS *op.Scope, index tf.Output) (output tf.Output) {
			seed := op.Pack(subS, []tf.Output{
				generation,
				op.Cast(subS, index, tf.Int64),
			})
			oneNoise := op.StatelessRandomNormal(subS, inputShape, seed, op.StatelessRandomNormalDtype(input.DataType()))
			stdev := op.Cast(subS.SubScope("stdev"), op.Const(subS.SubScope("stdev"), stdevVal), input.DataType())
			noise := op.Mul(subS, oneNoise, stdev)
			output = op.Add(subS, input, noise)
			return
		}
	}
}

// makeInt makes a PerturbFunc which perturbs ints.
func makeInt() es.PerturbFunc {
	return func(s *op.Scope, input tf.Output, numOutputs tf.Output, generation tf.Output, globalSeed int64) func(*op.Scope, tf.Output) tf.Output {
		inputShape := op.Shape(s.SubScope("input"), input, op.ShapeOutType(tf.Int32))
		return func(subS *op.Scope, index tf.Output) (output tf.Output) {
			seed := op.Pack(subS, []tf.Output{
				generation,
				op.Cast(subS, index, tf.Int64),
			})
			randomFloat := op.StatelessRandomNormal(subS, inputShape, seed, op.StatelessRandomNormalDtype(tf.Half))
			noise := op.Cast(subS.SubScope("noise"), randomFloat, input.DataType())
			output = op.Add(subS, input, noise)
			return
		}
	}
}

// Inc is a PerturbFunc which increments the input by one.
func Inc(s *op.Scope, input tf.Output, numOutputs tf.Output, generation tf.Output, globalSeed int64) func(*op.Scope, tf.Output) tf.Output {
	return func(subS *op.Scope, index tf.Output) (output tf.Output) {
		inc := op.Cast(subS, index, input.DataType())
		output = op.Add(subS, input, inc)
		return
	}
}

// Nil is a PerturbFunc which does nothing.
func Nil(s *op.Scope, input tf.Output, numOutputs tf.Output, generation tf.Output, globalSeed int64) func(*op.Scope, tf.Output) tf.Output {
	return func(subS *op.Scope, index tf.Output) (output tf.Output) {
		output = input
		return
	}
}
