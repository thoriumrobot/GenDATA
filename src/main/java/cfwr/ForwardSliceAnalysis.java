package cfwr;

import soot.*;
import soot.toolkits.graph.*;
import soot.toolkits.scalar.*;
import soot.jimple.*;
import soot.jimple.toolkits.pointer.*;

import java.util.*;

/**
 * Forward slicing analysis implementation using Soot's ForwardFlowAnalysis.
 * Performs forward slicing to find all statements that are influenced by a given slicing criterion.
 */
public class ForwardSliceAnalysis extends ForwardFlowAnalysis<Unit, FlowSet<Unit>> {
    
    private final Unit slicingCriterion;
    private final Set<Unit> relevantUnits;
    private final Map<Unit, Set<Local>> defsAtUnit;
    private final Map<Unit, Set<Local>> usesAtUnit;
    private final UnitGraph graph;
    
    public ForwardSliceAnalysis(UnitGraph graph, Unit slicingCriterion) {
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
        
        // Check if this unit should be included in the forward slice
        if (shouldIncludeUnit(in, unit)) {
            out.add(unit);
            
            // Add units that are data-dependent on this unit
            addDataDependentUnits(unit, out);
            
            // Add units that are control-dependent on this unit
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
        // Include unit if it's in the input set or if it's data/control dependent
        if (in.contains(unit)) {
            return true;
        }
        
        // Check for data dependencies
        Set<Local> uses = usesAtUnit.get(unit);
        if (uses != null) {
            for (Unit inUnit : in) {
                Set<Local> inDefs = defsAtUnit.get(inUnit);
                if (inDefs != null && !Collections.disjoint(uses, inDefs)) {
                    return true;
                }
            }
        }
        
        return false;
    }
    
    private void addDataDependentUnits(Unit unit, FlowSet<Unit> out) {
        // Find units that use variables defined by this unit
        Set<Local> defs = defsAtUnit.get(unit);
        if (defs == null || defs.isEmpty()) {
            return;
        }
        
        for (Unit otherUnit : graph) {
            if (otherUnit == unit) continue;
            
            Set<Local> uses = usesAtUnit.get(otherUnit);
            if (uses != null && !Collections.disjoint(defs, uses)) {
                out.add(otherUnit);
            }
        }
    }
    
    private void addControlDependentUnits(Unit unit, FlowSet<Unit> out) {
        // Find units that are control-dependent on this unit
        // This is a simplified implementation - a full implementation would
        // compute the post-dominator tree and identify control dependencies
        
        if (unit instanceof IfStmt) {
            // Add units in the true and false branches
            IfStmt ifStmt = (IfStmt) unit;
            addUnitsInBranch(ifStmt.getTarget(), out);
        }
    }
    
    private void addUnitsInBranch(Unit branchTarget, FlowSet<Unit> out) {
        // Add units in a branch (simplified implementation)
        // In practice, this would traverse the CFG to find all units in the branch
        if (branchTarget != null) {
            out.add(branchTarget);
        }
    }
    
    private void collectRelevantUnits() {
        // Collect all units that appear in any flow set
        for (Unit unit : graph) {
            FlowSet<Unit> flowSet = getFlowAfter(unit);
            if (flowSet != null && flowSet.contains(unit)) {
                relevantUnits.add(unit);
            }
        }
    }
    
    public Set<Unit> getForwardSlice() {
        return new HashSet<>(relevantUnits);
    }
    
    public boolean isRelevant(Unit unit) {
        return relevantUnits.contains(unit);
    }
}
