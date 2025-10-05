package cfwr;

import soot.*;
import soot.toolkits.graph.*;
import soot.toolkits.scalar.*;
import soot.jimple.*;
import soot.jimple.toolkits.pointer.*;

import java.util.*;

/**
 * Backward slicing analysis implementation using Soot's BackwardFlowAnalysis.
 * Performs backward slicing to find all statements that influence a given slicing criterion.
 */
public class BackwardSliceAnalysis extends BackwardFlowAnalysis<Unit, FlowSet<Unit>> {
    
    private final Unit slicingCriterion;
    private final Set<Unit> relevantUnits;
    private final Map<Unit, Set<Local>> defsAtUnit;
    private final Map<Unit, Set<Local>> usesAtUnit;
    private final UnitGraph graph;
    
    public BackwardSliceAnalysis(UnitGraph graph, Unit slicingCriterion) {
        super(graph);
        this.graph = graph;
        this.slicingCriterion = slicingCriterion;
        this.relevantUnits = new HashSet<>();
        this.defsAtUnit = new HashMap<>();
        this.usesAtUnit = new HashMap<>();
        
        // Initialize def-use information
        initializeDefUseInfo(graph);
        
        doAnalysis();
        
        // Collect all relevant units from the analysis
        collectRelevantUnits();
    }
    
    @Override
    protected FlowSet<Unit> newInitialFlow() {
        return new ArraySparseSet<>();
    }
    
    @Override
    protected FlowSet<Unit> entryInitialFlow() {
        FlowSet<Unit> entrySet = new ArraySparseSet<>();
        // Start with the slicing criterion
        entrySet.add(slicingCriterion);
        return entrySet;
    }
    
    @Override
    protected void flowThrough(FlowSet<Unit> in, Unit unit, FlowSet<Unit> out) {
        // Copy input to output
        out.clear();
        out.union(in);
        
        // Check if this unit should be included in the backward slice
        if (shouldIncludeUnit(in, unit)) {
            out.add(unit);
            
            // Add units that this unit is data-dependent on
            addDataDependentUnits(unit, out);
            
            // Add units that this unit is control-dependent on
            addControlDependentUnits(unit, out);
        }
    }
    
    @Override
    protected void merge(FlowSet<Unit> in1, FlowSet<Unit> in2, FlowSet<Unit> out) {
        out.clear();
        out.union(in1);
        out.union(in2);
    }
    
    @Override
    protected void copy(FlowSet<Unit> source, FlowSet<Unit> dest) {
        dest.clear();
        dest.union(source);
    }
    
    private void initializeDefUseInfo(UnitGraph graph) {
        // Build def-use information for all units
        for (Unit unit : graph) {
            Set<Local> defs = new HashSet<>();
            Set<Local> uses = new HashSet<>();
            
            // Analyze the unit to find defs and uses
            analyzeUnitForDefsUses(unit, defs, uses);
            
            defsAtUnit.put(unit, defs);
            usesAtUnit.put(unit, uses);
        }
    }
    
    private void analyzeUnitForDefsUses(Unit unit, Set<Local> defs, Set<Local> uses) {
        // Handle different types of statements
        if (unit instanceof DefinitionStmt) {
            DefinitionStmt defStmt = (DefinitionStmt) unit;
            
            // Add definition
            if (defStmt.getLeftOp() instanceof Local) {
                defs.add((Local) defStmt.getLeftOp());
            }
            
            // Add uses from right side
            collectUsesFromValue(defStmt.getRightOp(), uses);
        } else if (unit instanceof InvokeStmt) {
            InvokeStmt invokeStmt = (InvokeStmt) unit;
            collectUsesFromValue(invokeStmt.getInvokeExpr(), uses);
        } else if (unit instanceof ReturnStmt) {
            ReturnStmt returnStmt = (ReturnStmt) unit;
            if (returnStmt.getOp() != null) {
                collectUsesFromValue(returnStmt.getOp(), uses);
            }
        } else if (unit instanceof IfStmt) {
            IfStmt ifStmt = (IfStmt) unit;
            collectUsesFromValue(ifStmt.getCondition(), uses);
        }
        // Add more statement types as needed
    }
    
    private void collectUsesFromValue(Value value, Set<Local> uses) {
        if (value instanceof Local) {
            uses.add((Local) value);
        } else if (value instanceof BinopExpr) {
            BinopExpr binop = (BinopExpr) value;
            collectUsesFromValue(binop.getOp1(), uses);
            collectUsesFromValue(binop.getOp2(), uses);
        } else if (value instanceof UnopExpr) {
            UnopExpr unop = (UnopExpr) value;
            collectUsesFromValue(unop.getOp(), uses);
        } else if (value instanceof InvokeExpr) {
            InvokeExpr invoke = (InvokeExpr) value;
            for (Value arg : invoke.getArgs()) {
                collectUsesFromValue(arg, uses);
            }
            if (invoke instanceof InstanceInvokeExpr) {
                collectUsesFromValue(((InstanceInvokeExpr) invoke).getBase(), uses);
            }
        }
        // Add more value types as needed
    }
    
    private boolean shouldIncludeUnit(FlowSet<Unit> in, Unit unit) {
        // Include unit if it's in the input set or if it influences units in the set
        if (in.contains(unit)) {
            return true;
        }
        
        // Check for data dependencies (this unit defines variables used by units in the input set)
        Set<Local> defs = defsAtUnit.get(unit);
        if (defs != null && !defs.isEmpty()) {
            for (Unit inUnit : in) {
                Set<Local> inUses = usesAtUnit.get(inUnit);
                if (inUses != null && !Collections.disjoint(defs, inUses)) {
                    return true;
                }
            }
        }
        
        return false;
    }
    
    private void addDataDependentUnits(Unit unit, FlowSet<Unit> out) {
        // Find units that define variables used by this unit
        Set<Local> uses = usesAtUnit.get(unit);
        if (uses == null || uses.isEmpty()) {
            return;
        }
        
        for (Unit otherUnit : graph) {
            if (otherUnit == unit) continue;
            
            Set<Local> defs = defsAtUnit.get(otherUnit);
            if (defs != null && !Collections.disjoint(defs, uses)) {
                out.add(otherUnit);
            }
        }
    }
    
    private void addControlDependentUnits(Unit unit, FlowSet<Unit> out) {
        // Find units that control this unit (simplified implementation)
        // In practice, this would compute the post-dominator tree
        
        if (unit instanceof IfStmt) {
            // Add the condition that controls this if statement
            IfStmt ifStmt = (IfStmt) unit;
            // The condition is already handled in analyzeUnitForDefsUses
        }
        
        // Add predecessor units that might control this unit
        for (Unit pred : graph.getPredsOf(unit)) {
            if (pred instanceof IfStmt) {
                out.add(pred);
            }
        }
    }
    
    private void collectRelevantUnits() {
        // Collect all units that appear in any flow set
        for (Unit unit : graph) {
            FlowSet<Unit> flowSet = getFlowBefore(unit);
            if (flowSet != null && flowSet.contains(unit)) {
                relevantUnits.add(unit);
            }
        }
    }
    
    public Set<Unit> getBackwardSlice() {
        return new HashSet<>(relevantUnits);
    }
    
    public boolean isRelevant(Unit unit) {
        return relevantUnits.contains(unit);
    }
}
