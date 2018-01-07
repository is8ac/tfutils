package es

import (
	"strconv"

	"github.com/is8ac/tfutils/tb"
	tf "github.com/tensorflow/tensorflow/tensorflow/go"
	"github.com/tensorflow/tensorflow/tensorflow/go/op"
)

// PerturbFunc takes an output and a base seed, and returns a function which returns perturbed version of the output
type PerturbFunc func(s *op.Scope, input tf.Output, numOutputs tf.Output, generation tf.Output, globalSeed int64) func(*op.Scope, tf.Output) tf.Output

// VarDef defines a shape and name
type VarDef struct {
	Name        string      // Name must be unique
	DataType    tf.DataType // DataType, for now just tf.Float
	Shape       tf.Shape    // Shape is the shape of the var
	PerturbFunc PerturbFunc // func to perturb the var
}

// ModelDef defines a model
type ModelDef struct {
	VarDefs []VarDef                                          // The list of shapes of vars that the child wants
	Model   func(*op.Scope, []tf.Output, tf.Output) tf.Output // Child takes a slice of vars, some inputs and outputs, and returns a loss.
}

// SingleLayerNN create a modelDef for a single layer nn.
func SingleLayerNN(inputSize, outputSize int64) (model ModelDef) {
	model.VarDefs = []VarDef{
		VarDef{Name: "weights", DataType: tf.Float, Shape: tf.MakeShape(inputSize, outputSize), PerturbFunc: MakeSlicePerturb(0.003)},
		VarDef{Name: "biases", DataType: tf.Float, Shape: tf.MakeShape(outputSize), PerturbFunc: MakeSlicePerturb(0.003)},
	}

	model.Model = func(s *op.Scope, vars []tf.Output, images tf.Output) (output tf.Output) {
		output = op.Add(s, vars[1], op.MatMul(s, images, vars[0]))
		return
	}
	return
}

// DummyModel is dummy
func DummyModel(outputSize int64) (model ModelDef) {
	model.VarDefs = []VarDef{
		VarDef{Name: "biases", DataType: tf.Float, Shape: tf.MakeShape(outputSize), PerturbFunc: IncPerturb},
	}
	model.Model = func(s *op.Scope, vars []tf.Output, images tf.Output) (output tf.Output) {
		output = op.Tile(s, op.ExpandDims(s, vars[0], op.Const(s.SubScope("dims"), int32(0))), op.Const(s, []int32{10000, 1}))
		return
	}
	return
}

func varCache(s *op.Scope, input tf.Output, name string) (init *tf.Operation, output tf.Output) {
	variable := op.VarHandleOp(s, input.DataType(), input.Shape(), op.VarHandleOpSharedName(name))
	init = op.AssignVariableOp(s, variable, input)
	output = op.ReadVariableOp(s, variable, input.DataType())
	return
}

// MakeSlicePerturb makes a PerturbFunc which uses a slice of a single tensor. Better performance.
func MakeSlicePerturb(stdev float32) PerturbFunc {
	return func(s *op.Scope, input tf.Output, numOutputs tf.Output, generation tf.Output, globalSeed int64) func(*op.Scope, tf.Output) tf.Output {
		inputShape := op.Shape(s.SubScope("input"), input, op.ShapeOutType(tf.Int32))
		noiseShape := op.Add(s.SubScope("shape"), inputShape, numOutputs)
		seed := op.Pack(s, []tf.Output{generation, op.Const(s.SubScope("seed"), globalSeed)})
		oneNoise := op.StatelessRandomNormal(s, noiseShape, seed)
		noise := op.Mul(s, oneNoise, op.Const(s.SubScope("stdev"), stdev))
		return func(subS *op.Scope, index tf.Output) (output tf.Output) {
			begin := op.Fill(subS.SubScope("size"), op.Shape(subS.SubScope("dims"), inputShape, op.ShapeOutType(tf.Int32)), index)
			noiseSlice := op.Slice(subS, noise, begin, inputShape)
			output = op.Add(subS, input, noiseSlice)
			return
		}
	}
}

// MakeSimplePerturb is a PerturbFunc which uses generation and index as seeds.
func MakeSimplePerturb(stdev float32) PerturbFunc {
	return func(s *op.Scope, input tf.Output, numOutputs tf.Output, generation tf.Output, globalSeed int64) func(*op.Scope, tf.Output) tf.Output {
		inputShape := op.Shape(s.SubScope("input"), input, op.ShapeOutType(tf.Int32))
		return func(subS *op.Scope, index tf.Output) (output tf.Output) {
			seed := op.Pack(subS, []tf.Output{
				generation,
				op.Cast(subS, index, tf.Int64),
			})
			noise := op.StatelessRandomNormal(subS, inputShape, seed)
			output = op.Add(subS, input, noise)
			return
		}
	}
}

// IncPerturb is a PerturbFunc which increments the input by one.
func IncPerturb(s *op.Scope, input tf.Output, numOutputs tf.Output, generation tf.Output, globalSeed int64) func(*op.Scope, tf.Output) tf.Output {
	return func(subS *op.Scope, index tf.Output) (output tf.Output) {
		inc := op.Cast(subS, index, tf.Float)
		output = op.Add(subS, input, inc)
		return
	}
}

// ESsess holds a training session.
type ESsess struct {
	sess                *tf.Session
	graph               *tf.Graph
	bestIndex           tf.Output
	updateOPs           []*tf.Operation
	indexes             []int32
	indexPH             tf.Output
	perturbFromIndex    []*tf.Operation
	readGeneration      tf.Output
	readVars            []tf.Output
	incrementGeneration *tf.Operation
	accuracy            tf.Output
}

// WriteTBgraph writes the graph to tensorboard
func (sess ESsess) WriteTBgraph(path string) (err error) {
	err = tb.WriteGraphSummary(sess.graph, path)
	return
}

// Perturb updates the vars with the given seed. Is deterministic if seed is non 0.
func (sess ESsess) Perturb(index int32) (err error) {
	indexTensor, err := tf.NewTensor(index)
	if err != nil {
		return
	}
	_, err = sess.sess.Run(
		map[tf.Output]*tf.Tensor{sess.indexPH: indexTensor},
		nil,
		sess.perturbFromIndex,
	)
	if err != nil {
		panic(err)
	}
	// we need to increment the generation so that it won't indeterministically interfere with the var updates.
	_, err = sess.sess.Run(
		nil,
		nil,
		[]*tf.Operation{sess.incrementGeneration},
	)
	return
}

// ReadGeneration reads the value of the generation var from the graph.
func (sess ESsess) ReadGeneration() (generation int64, err error) {
	results, err := sess.sess.Run(
		nil,
		[]tf.Output{sess.readGeneration},
		nil,
	)
	generation = results[0].Value().(int64)
	return
}

// ReadVars reads the value of the vars var from the graph.
func (sess ESsess) ReadVars() (vars []*tf.Tensor, err error) {
	vars, err = sess.sess.Run(
		nil,
		sess.readVars,
		nil,
	)
	return
}

// Freeze freezes a model with the current vars.
func (sess ESsess) Freeze(s *op.Scope, input tf.Output) (result tf.Output) {
	panic("not implemented")
	return
}

// BestIndex each index and returns the index of the best.
func (sess ESsess) BestIndex() (bestIndex int32) {
	results, err := sess.sess.Run(
		map[tf.Output]*tf.Tensor{},
		[]tf.Output{sess.bestIndex},
		nil,
	)
	if err != nil {
		panic(err)
	}
	bestIndex = results[0].Value().(int32)
	return
}

// Accuracy on the test set.
func (sess ESsess) Accuracy() (accuracy float32) {
	results, err := sess.sess.Run(
		map[tf.Output]*tf.Tensor{},
		[]tf.Output{sess.accuracy},
		nil,
	)
	if err != nil {
		panic(err)
	}
	accuracy = results[0].Value().(float32)
	return
}

// Accuracy calculates the accuracy of predictions
type Accuracy func(s *op.Scope, actual, target tf.Output) (accuracy tf.Output)

// PercentAccuracy calculates the percent of the predictions whoes top matches the targets.
func PercentAccuracy(s *op.Scope, actual, target tf.Output) (accuracy tf.Output) {
	actualLabels := op.Cast(s.SubScope("actual"), op.ArgMax(s, actual, op.Const(s.SubScope("argmax_dim"), int32(1)), op.ArgMaxOutputType(tf.Int32)), tf.Uint8)
	correct := op.Equal(s, actualLabels, target)
	accuracy = op.Mean(s, op.Cast(s.SubScope("accuracy"), correct, tf.Float), op.Const(s.SubScope("mean_dim"), int32(0)))
	return
}

// Loss takes an actual and a target and returns a loss
type Loss func(s *op.Scope, actual, target tf.Output) (loss tf.Output)

func softmaxSqrDifLoss(s *op.Scope, actual, target tf.Output) (loss tf.Output) {
	softmax := op.Softmax(s, actual)
	sqrDiffs := op.SquaredDifference(s, softmax, target)
	sums := op.Sum(s, sqrDiffs, op.Const(s, int32(1)))
	loss = op.Mean(s, sums, op.Const(s.SubScope("mean_reduce_dims"), []int32{0}))
	return
}

// NewSession creates a new ESsess. Takes ownership of and finalises scope.
func NewSession(s *op.Scope, md ModelDef, lossFunc Loss, accuracyFunc Accuracy, inputs, targets, testInputs, testTargets tf.Output, initOPs []*tf.Operation, numChildren int, globalSeed int64) (esSess ESsess, err error) {
	esSess.indexPH = op.Placeholder(s.SubScope("index"), tf.Int32, op.PlaceholderShape(tf.ScalarShape())) // to be filled with the child index
	generationScope := s.SubScope("generation")
	generation := op.VarHandleOp(generationScope, tf.Int64, tf.ScalarShape(), op.VarHandleOpSharedName("generation"))                     // stores the generation
	initGeneration := op.AssignVariableOp(s.SubScope("init_generation"), generation, op.Const(s.SubScope("zero"), int64(0)))              // inits generation to 0
	esSess.readGeneration = op.ReadVariableOp(generationScope, generation, tf.Int64)                                                      // read the generation var
	esSess.incrementGeneration = op.AssignAddVariableOp(generationScope, generation, op.Const(generationScope.SubScope("one"), int64(1))) // increment the generation

	// Now we create slices to hold various things for each var.
	varCount := len(md.VarDefs)                                            // number of vars
	vars := make([]tf.Output, varCount)                                    // handles to the actual variables
	esSess.readVars = make([]tf.Output, varCount)                          // outputs to read the value of the variables
	initVars := make([]*tf.Operation, varCount)                            // operations to initialise the variables with zeros
	esSess.perturbFromIndex = make([]*tf.Operation, varCount)              // for perturbing according to a given index.
	perturbFuncs := make([]func(*op.Scope, tf.Output) tf.Output, varCount) // funcs to perturb vars

	zero := op.Const(s.SubScope("zero"), float32(0)) // a single float32 0 used to fill

	numSlices := op.Const(s.SubScope("num_slices"), int32(numChildren))
	// for each parameter
	for i, vd := range md.VarDefs {
		varScope := s.SubScope(vd.Name)
		sliceShape, err := vd.Shape.ToSlice()
		if err != nil {
			panic(err)
		}
		zeroVar := op.Fill(varScope, op.Cast(varScope, op.Const(varScope, sliceShape), tf.Int32), zero)
		vars[i] = op.VarHandleOp(varScope, vd.DataType, vd.Shape, op.VarHandleOpSharedName("foo_"+vd.Name))
		initVars[i] = op.AssignVariableOp(varScope, vars[i], zeroVar)
		esSess.readVars[i] = op.ReadVariableOp(varScope, vars[i], vd.DataType)
		perturbFuncs[i] = vd.PerturbFunc(varScope.SubScope("tmp_perturb"), esSess.readVars[i], numSlices, esSess.readGeneration, globalSeed)
		perturbedFromIndex := vd.PerturbFunc(varScope.SubScope("init_index_perturb"), esSess.readVars[i], numSlices, esSess.readGeneration, globalSeed)(varScope.SubScope("index_perturb"), esSess.indexPH)
		esSess.perturbFromIndex[i] = op.AssignVariableOp(s.SubScope("perturb_index"), vars[i], perturbedFromIndex)
	}

	lossList := make([]tf.Output, numChildren)
	// for each child,
	for childIndex := range lossList {
		childScope := s.SubScope("child" + strconv.Itoa(childIndex))
		index := op.Const(childScope.SubScope("index"), int32(childIndex))
		perturbedVars := make([]tf.Output, len(perturbFuncs))
		// for each var,
		for i, varDef := range md.VarDefs {
			perturbedVars[i] = perturbFuncs[i](childScope.SubScope(varDef.Name), index) // get the perturbFunc for that vars, and run it for that index
		}
		actual := md.Model(childScope.SubScope("model"), perturbedVars, inputs)
		lossList[childIndex] = lossFunc(childScope.SubScope("loss"), actual, targets)
	}

	lossPop := op.Pack(s, lossList)
	esSess.bestIndex = op.ArgMin(s, lossPop, op.Const(s.SubScope("argmin_dims"), int32(0)), op.ArgMinOutputType(tf.Int32))

	// accurecy
	testScope := s.SubScope("test")
	testActual := md.Model(testScope.SubScope("model"), esSess.readVars, testInputs)
	esSess.accuracy = accuracyFunc(testScope.SubScope("accurecy"), testActual, testTargets)
	esSess.graph, err = s.Finalize()
	if err != nil {
		return
	}
	esSess.sess, err = tf.NewSession(esSess.graph, nil)
	if err != nil {
		return
	}
	_, err = esSess.sess.Run(nil, nil, append(append(initVars, initGeneration), initOPs...))
	if err != nil {
		return
	}
	return
}
