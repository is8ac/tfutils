// Package es provides tools to optimize the parameters of arbitrary tensorflow graphs via evolutionary strategy.
package es

import (
	"strconv"

	"github.com/is8ac/tfutils/tb"
	tf "github.com/tensorflow/tensorflow/tensorflow/go"
	"github.com/tensorflow/tensorflow/tensorflow/go/op"
)

//IDEA: Average perturbations weighted by loss.

// PerturbFunc takes an output and a base seed, and returns a function which returns perturbed version of the output
type PerturbFunc func(s *op.Scope, input tf.Output, numOutputs tf.Output, generation tf.Output, globalSeed int64) func(*op.Scope, tf.Output) tf.Output

// ParamDef defines a shape and name
type ParamDef struct {
	Name        string                    // Name must be unique
	ZeroVal     func(*op.Scope) tf.Output // func to make a scalar zero of whatever data Type
	Shape       tf.Shape                  // Shape is the shape of the var
	PerturbFunc PerturbFunc               // func to perturb the var
}

// ModelDef defines a model
type ModelDef struct {
	ParamDefs []ParamDef                                                                                   // The list of shapes of vars that the child wants
	Model     func(s *op.Scope, params []tf.Output, inputs tf.Output) (output tf.Output, tbOPs []tb.LogOP) // Child takes a slice of vars, some inputs and outputs, and returns a loss.
}

// ESsess holds a training session.
type ESsess struct {
	sess                  *tf.Session
	graph                 *tf.Graph
	bestIndex             tf.Output
	updateOPs             []*tf.Operation
	indexes               []int32
	indexPH               tf.Output
	perturbFromIndex      []*tf.Operation
	readGeneration        tf.Output
	readVars              []tf.Output
	incrementGeneration   *tf.Operation
	accuracy              tf.Output
	modelDef              ModelDef
	modelSummaryWriterOps []*tf.Operation
	accuracySummaryWriter *tf.Operation
	closeSummaryWriter    *tf.Operation
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

// ReadParams reads the value of the params variables from the graph.
func (sess ESsess) ReadParams() (vars []*tf.Tensor, err error) {
	vars, err = sess.sess.Run(
		nil,
		sess.readVars,
		nil,
	)
	return
}

// Freeze freezes a model with the current vars.
func (sess ESsess) Freeze(s *op.Scope, input tf.Output) (output tf.Output, err error) {
	params, err := sess.ReadParams()
	if err != nil {
		return
	}
	if len(params) != len(sess.modelDef.ParamDefs) {
		panic("len(params) != len(sess.modelDef.ParamDefs), This should never happen!!")
	}
	consts := make([]tf.Output, len(params))
	for i, paramTensor := range params {
		consts[i] = op.Const(s.SubScope(sess.modelDef.ParamDefs[i].Name), paramTensor)
	}
	output, _ = sess.modelDef.Model(s, consts, input)
	return
}

// BestIndex each index and returns the index of the best.
func (sess ESsess) BestIndex() (bestIndex int32, err error) {
	results, err := sess.sess.Run(
		map[tf.Output]*tf.Tensor{},
		[]tf.Output{sess.bestIndex},
		nil,
	)
	if err != nil {
		return
	}
	bestIndex = results[0].Value().(int32)
	return
}

// Accuracy on the test set.
func (sess ESsess) Accuracy(logAcc, logModel bool) (accuracy float32, err error) {
	var ops []*tf.Operation
	if logAcc {
		ops = append(ops, sess.accuracySummaryWriter)
	}
	if logModel {
		ops = append(ops, sess.modelSummaryWriterOps...)
	}
	results, err := sess.sess.Run(
		map[tf.Output]*tf.Tensor{},
		[]tf.Output{sess.accuracy},
		ops,
	)
	if err != nil {
		return
	}
	accuracy = results[0].Value().(float32)
	return
}

// Close the tf session and tensorboard logging cleanly.
func (sess ESsess) Close() (err error) {
	_, err = sess.sess.Run(
		nil,
		nil,
		[]*tf.Operation{sess.closeSummaryWriter},
	)
	if err != nil {
		return
	}
	//err = sess.Close()
	return
}

// Accuracy calculates the accuracy of predictions
type Accuracy func(s *op.Scope, actual, target tf.Output) (accuracy tf.Output)

// Loss takes an actual and a target and returns a loss
type Loss func(s *op.Scope, actual, target tf.Output) (loss tf.Output)

// NewSession creates a new ESsess. Takes ownership of and finalises scope.
func NewSession(s *op.Scope,
	md ModelDef,
	lossFunc Loss,
	accuracyFunc Accuracy,
	inputs, targets, testInputs, testTargets tf.Output,
	initOPs []*tf.Operation,
	numChildren int,
	globalSeed int64,
	tensorboardLogDir string,
	runName string,
) (esSess ESsess, err error) {
	esSess.modelDef = md
	esSess.indexPH = op.Placeholder(s.SubScope("index"), tf.Int32, op.PlaceholderShape(tf.ScalarShape())) // to be filled with the child index
	generationScope := s.SubScope("generation")
	generation := op.VarHandleOp(generationScope, tf.Int64, tf.ScalarShape(), op.VarHandleOpSharedName("generation"))                     // stores the generation
	initGeneration := op.AssignVariableOp(s.SubScope("init_generation"), generation, op.Const(s.SubScope("zero"), int64(0)))              // inits generation to 0
	esSess.readGeneration = op.ReadVariableOp(generationScope, generation, tf.Int64)                                                      // read the generation var
	esSess.incrementGeneration = op.AssignAddVariableOp(generationScope, generation, op.Const(generationScope.SubScope("one"), int64(1))) // increment the generation

	// Now we create slices to hold various things for each var.
	varCount := len(md.ParamDefs)                                          // number of vars
	vars := make([]tf.Output, varCount)                                    // handles to the actual variables
	esSess.readVars = make([]tf.Output, varCount)                          // outputs to read the value of the variables
	initVars := make([]*tf.Operation, varCount)                            // operations to initialise the variables with zeros
	esSess.perturbFromIndex = make([]*tf.Operation, varCount)              // for perturbing according to a given index.
	perturbFuncs := make([]func(*op.Scope, tf.Output) tf.Output, varCount) // funcs to perturb vars

	numSlices := op.Const(s.SubScope("num_slices"), int32(numChildren))
	// for each parameter
	for i, vd := range md.ParamDefs {
		varScope := s.SubScope(vd.Name)
		zero := vd.ZeroVal(varScope.SubScope("zero"))
		sliceShape, err := vd.Shape.ToSlice()
		if err != nil {
			panic(err)
		}
		zeroVar := op.Fill(varScope, op.Cast(varScope, op.Const(varScope, sliceShape), tf.Int32), zero)
		vars[i] = op.VarHandleOp(varScope, zero.DataType(), vd.Shape, op.VarHandleOpSharedName("foo_"+vd.Name))
		initVars[i] = op.AssignVariableOp(varScope, vars[i], zeroVar)
		esSess.readVars[i] = op.ReadVariableOp(varScope, vars[i], zero.DataType())
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
		for i, ParamDef := range md.ParamDefs {
			perturbedVars[i] = perturbFuncs[i](childScope.SubScope(ParamDef.Name), index) // get the perturbFunc for that vars, and run it for that index
		}
		actual, _ := md.Model(childScope.SubScope("model"), perturbedVars, inputs)
		lossList[childIndex] = lossFunc(childScope.SubScope("loss"), actual, targets)
	}

	lossPop := op.Pack(s, lossList)
	esSess.bestIndex = op.ArgMin(s, lossPop, op.Const(s.SubScope("argmin_dims"), int32(0)), op.ArgMinOutputType(tf.Int32))

	// accuracy
	writer := op.SummaryWriter(s, op.SummaryWriterSharedName("tb_logs"))
	createSummaryWriter := op.CreateSummaryFileWriter(s,
		writer,
		op.Const(s.SubScope("log_dir"), tensorboardLogDir+"/"+runName),
		op.Const(s.SubScope("max_queue"), int32(10)),
		op.Const(s.SubScope("flush_millis"), int32(100)),
		op.Const(s.SubScope("filename_suffix"), ".tblog"),
	)
	esSess.closeSummaryWriter = op.CloseSummaryWriter(s, writer)
	tbs := s.SubScope("summaries")

	testScope := s.SubScope("test")
	testActual, logOPs := md.Model(testScope.SubScope("model"), esSess.readVars, testInputs)
	esSess.accuracy = accuracyFunc(testScope.SubScope("accurecy"), testActual, testTargets)
	accuracyTag := op.Const(testScope.SubScope("summary_tag"), "accuracy")
	esSess.accuracySummaryWriter = op.WriteScalarSummary(testScope, writer, esSess.readGeneration, accuracyTag, esSess.accuracy)

	esSess.modelSummaryWriterOps = make([]*tf.Operation, len(logOPs))
	for i, logOP := range logOPs {
		ss := tbs.SubScope(logOP.Name)                                                         // create a sub scope for this summary.
		tag := op.Const(ss, logOP.Name)                                                        // create a tag for the summary.
		esSess.modelSummaryWriterOps[i] = logOP.OPfunc(ss, writer, tag, esSess.readGeneration) // call the func which the modelDef gave us to get the summary writer operation.
	}
	esSess.graph, err = s.Finalize()
	if err != nil {
		return
	}
	esSess.sess, err = tf.NewSession(esSess.graph, nil)
	if err != nil {
		return
	}
	_, err = esSess.sess.Run(nil, nil, append(append(initVars, initGeneration, createSummaryWriter), initOPs...))
	if err != nil {
		return
	}
	return
}
