/*
    @Positive
 * Copyright (c) 1999, 2021, Oracle and/or its affiliates. All rights reserved.
    @Positive
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
    @Positive
 *
    @Positive
 * This code is free software; you can redistribute it and/or modify it
    @Positive
 * under the terms of the GNU General Public License version 2 only, as
    @Positive
 * published by the Free Software Foundation.  Oracle designates this
    @Positive
 * particular file as subject to the "Classpath" exception as provided
    @Positive
 * by Oracle in the LICENSE file that accompanied this code.
    @Positive
 *
    @Positive
 * This code is distributed in the hope that it will be useful, but WITHOUT
    @Positive
 * ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
    @Positive
 * FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public License
    @Positive
 * version 2 for more details (a copy is included in the LICENSE file that
    @Positive
 * accompanied this code).
    @Positive
 *
    @Positive
 * You should have received a copy of the GNU General Public License version
    @Positive
 * 2 along with this work; if not, write to the Free Software Foundation,
    @Positive
 * Inc., 51 Franklin St, Fifth Floor, Boston, MA 02110-1301 USA.
    @Positive
 *
    @Positive
 * Please contact Oracle, 500 Oracle Parkway, Redwood Shores, CA 94065 USA
    @Positive
 * or visit www.oracle.com if you need additional information or have any
    @Positive
 * questions.
    @Positive
 */
    @Positive
package com.sun.tools.javac.comp;

    @Positive
import org.checkerframework.dataflow.qual.Pure;
    @Positive
import com.sun.tools.javac.api.Formattable.LocalizedString;
    @Positive
import com.sun.tools.javac.code.*;
    @Positive
import com.sun.tools.javac.code.Scope.WriteableScope;
    @Positive
import com.sun.tools.javac.code.Source.Feature;
    @Positive
import com.sun.tools.javac.code.Symbol.*;
    @Positive
import com.sun.tools.javac.code.Type.*;
    @Positive
import com.sun.tools.javac.comp.Attr.ResultInfo;
    @Positive
import com.sun.tools.javac.comp.Check.CheckContext;
    @Positive
import com.sun.tools.javac.comp.DeferredAttr.AttrMode;
    @Positive
import com.sun.tools.javac.comp.DeferredAttr.DeferredAttrContext;
    @Positive
import com.sun.tools.javac.comp.DeferredAttr.DeferredType;
    @Positive
import com.sun.tools.javac.comp.Resolve.MethodResolutionContext.Candidate;
    @Positive
import com.sun.tools.javac.comp.Resolve.MethodResolutionDiagHelper.Template;
    @Positive
import com.sun.tools.javac.comp.Resolve.ReferenceLookupResult.StaticKind;
    @Positive
import com.sun.tools.javac.jvm.*;
    @Positive
import com.sun.tools.javac.main.Option;
    @Positive
import com.sun.tools.javac.resources.CompilerProperties.Errors;
    @Positive
import com.sun.tools.javac.resources.CompilerProperties.Fragments;
    @Positive
import com.sun.tools.javac.resources.CompilerProperties.Warnings;
    @Positive
import com.sun.tools.javac.tree.*;
    @Positive
import com.sun.tools.javac.tree.JCTree.*;
    @Positive
import com.sun.tools.javac.tree.JCTree.JCMemberReference.ReferenceKind;
    @Positive
import com.sun.tools.javac.tree.JCTree.JCPolyExpression.*;
    @Positive
import com.sun.tools.javac.util.*;
    @Positive
import com.sun.tools.javac.util.DefinedBy.Api;
    @Positive
import com.sun.tools.javac.util.JCDiagnostic.DiagnosticFlag;
    @Positive
import com.sun.tools.javac.util.JCDiagnostic.DiagnosticPosition;
    @Positive
import com.sun.tools.javac.util.JCDiagnostic.DiagnosticType;
    @Positive
import com.sun.tools.javac.util.JCDiagnostic.Warning;
    @Positive
import java.util.Arrays;
    @Positive
import java.util.Collection;
    @Positive
import java.util.EnumSet;
    @Positive
import java.util.HashSet;
    @Positive
import java.util.Iterator;
    @Positive
import java.util.LinkedHashMap;
    @Positive
import java.util.Map;
    @Positive
import java.util.Set;
    @Positive
import java.util.function.BiFunction;
    @Positive
import java.util.function.BiPredicate;
    @Positive
import java.util.function.Consumer;
    @Positive
import java.util.function.Function;
    @Positive
import java.util.function.Predicate;
    @Positive
import java.util.stream.Stream;
    @Positive
import java.util.stream.StreamSupport;
    @Positive
import javax.lang.model.element.ElementVisitor;
    @Positive
import static com.sun.tools.javac.code.Flags.*;
    @Positive
import static com.sun.tools.javac.code.Flags.BLOCK;
    @Positive
import static com.sun.tools.javac.code.Flags.STATIC;
    @Positive
import static com.sun.tools.javac.code.Kinds.*;
    @Positive
import static com.sun.tools.javac.code.Kinds.Kind.*;
    @Positive
import static com.sun.tools.javac.code.TypeTag.*;
    @Positive
import static com.sun.tools.javac.comp.Resolve.MethodResolutionPhase.*;
    @Positive
import static com.sun.tools.javac.tree.JCTree.Tag.*;
    @Positive
import static com.sun.tools.javac.util.Iterators.createCompoundIterator;

    @Positive
public class Resolve {

    @Positive
    protected static final Context.Key<Resolve> resolveKey;

    @Positive
    public final boolean allowFunctionalInterfaceMostSpecific;

    @Positive
    public final boolean allowModules;

    @Positive
    public final boolean allowRecords;

    @Positive
    public final boolean checkVarargsAccessAfterResolution;

    @Positive
    protected Resolve(Context context) {
    @Positive
    }

    @Positive
    public static Resolve instance(Context context);

    @Positive
    void reportVerboseResolutionDiagnostic(DiagnosticPosition dpos, Name name, Type site, List<Type> argtypes, List<Type> typeargtypes, Symbol bestSoFar);

    @Positive
    JCDiagnostic getVerboseApplicableCandidateDiag(int pos, Symbol sym, Type inst);

    @Positive
    JCDiagnostic getVerboseInapplicableCandidateDiag(int pos, Symbol sym, JCDiagnostic subDiag);

    @Positive
    protected static boolean isStatic(Env<AttrContext> env);

    @Positive
    static boolean isInitializer(Env<AttrContext> env);

    @Positive
    public boolean isAccessible(Env<AttrContext> env, TypeSymbol c);

    @Positive
    public boolean isAccessible(Env<AttrContext> env, TypeSymbol c, boolean checkInner);

    @Positive
    boolean isAccessible(Env<AttrContext> env, Type t);

    @Positive
    boolean isAccessible(Env<AttrContext> env, Type t, boolean checkInner);

    @Positive
    public boolean isAccessible(Env<AttrContext> env, Type site, Symbol sym);

    @Positive
    public boolean isAccessible(Env<AttrContext> env, Type site, Symbol sym, boolean checkInner);

    @Positive
    void checkAccessibleType(Env<AttrContext> env, Type t);

    @Positive
    Type rawInstantiate(Env<AttrContext> env, Type site, Symbol m, ResultInfo resultInfo, List<Type> argtypes, List<Type> typeargtypes, boolean allowBoxing, boolean useVarargs, Warner warn) throws Infer.InferenceException;

    @Positive
    Type checkMethod(Env<AttrContext> env, Type site, Symbol m, ResultInfo resultInfo, List<Type> argtypes, List<Type> typeargtypes, Warner warn);

    @Positive
    Type instantiate(Env<AttrContext> env, Type site, Symbol m, ResultInfo resultInfo, List<Type> argtypes, List<Type> typeargtypes, boolean allowBoxing, boolean useVarargs, Warner warn);

    @Positive
    interface MethodCheck {

    @Positive
        void argumentsAcceptable(Env<AttrContext> env, DeferredAttrContext deferredAttrContext, List<Type> argtypes, List<Type> formals, Warner warn);

    @Positive
        MethodCheck mostSpecificCheck(List<Type> actuals);
    @Positive
    }

    @Positive
    abstract class AbstractMethodCheck implements MethodCheck {

    @Positive
        @Override
    @Positive
        public void argumentsAcceptable(final Env<AttrContext> env, DeferredAttrContext deferredAttrContext, List<Type> argtypes, List<Type> formals, Warner warn);

    @Positive
        abstract void checkArg(DiagnosticPosition pos, boolean varargs, Type actual, Type formal, DeferredAttrContext deferredAttrContext, Warner warn);

    @Positive
        protected void reportMC(DiagnosticPosition pos, MethodCheckDiag diag, InferenceContext inferenceContext, Object... args);

    @Positive
        class SharedInapplicableMethodException extends InapplicableMethodException {

    @Positive
            SharedInapplicableMethodException setMessage(JCDiagnostic details);
    @Positive
        }

    @Positive
        public MethodCheck mostSpecificCheck(List<Type> actuals);
    @Positive
    }

    @Positive
    class MethodReferenceCheck extends AbstractMethodCheck {

    @Positive
        @Override
    @Positive
        void checkArg(DiagnosticPosition pos, boolean varargs, Type actual, Type formal, DeferredAttrContext deferredAttrContext, Warner warn);

    @Positive
        @Override
    @Positive
        public MethodCheck mostSpecificCheck(List<Type> actuals);

    @Positive
        @Override
    @Positive
        public String toString();
    @Positive
    }

    @Positive
    abstract class MethodCheckContext implements CheckContext {

    @Positive
        public MethodCheckContext(boolean strict, DeferredAttrContext deferredAttrContext, Warner rsWarner) {
    @Positive
        }

    @Positive
        public boolean compatible(Type found, Type req, Warner warn);

    @Positive
        public void report(DiagnosticPosition pos, JCDiagnostic details);

    @Positive
        public Warner checkWarner(DiagnosticPosition pos, Type found, Type req);

    @Positive
        public InferenceContext inferenceContext();

    @Positive
        public DeferredAttrContext deferredAttrContext();

    @Positive
        @Override
    @Positive
        public String toString();
    @Positive
    }

    @Positive
    class MethodResultInfo extends ResultInfo {

    @Positive
        public MethodResultInfo(Type pt, CheckContext checkContext) {
    @Positive
        }

    @Positive
        @Override
    @Positive
        protected Type check(DiagnosticPosition pos, Type found);

    @Positive
        @Override
    @Positive
        protected MethodResultInfo dup(Type newPt);

    @Positive
        @Override
    @Positive
        protected ResultInfo dup(CheckContext newContext);

    @Positive
        @Override
    @Positive
        protected ResultInfo dup(Type newPt, CheckContext newContext);
    @Positive
    }

    @Positive
    class MostSpecificCheck implements MethodCheck {

    @Positive
        @Override
    @Positive
        public void argumentsAcceptable(final Env<AttrContext> env, DeferredAttrContext deferredAttrContext, List<Type> formals1, List<Type> formals2, Warner warn);

    @Positive
        ResultInfo methodCheckResult(Type to, DeferredAttr.DeferredAttrContext deferredAttrContext, Warner rsWarner, Type actual);

    @Positive
        class MostSpecificCheckContext extends MethodCheckContext {

    @Positive
            public MostSpecificCheckContext(DeferredAttrContext deferredAttrContext, Warner rsWarner, Type actual) {
    @Positive
            }

    @Positive
            public boolean compatible(Type found, Type req, Warner warn);

    @Positive
            class MostSpecificFunctionReturnChecker extends DeferredAttr.PolyScanner {

    @Positive
                @Override
    @Positive
                void skip(JCTree tree);

    @Positive
                @Override
    @Positive
                public void visitConditional(JCConditional tree);

    @Positive
                @Override
    @Positive
                public void visitReference(JCMemberReference tree);

    @Positive
                @Override
    @Positive
                public void visitParens(JCParens tree);

    @Positive
                @Override
    @Positive
                public void visitLambda(JCLambda tree);
    @Positive
            }
    @Positive
        }

    @Positive
        public MethodCheck mostSpecificCheck(List<Type> actuals);
    @Positive
    }

    @Positive
    public static class InapplicableMethodException extends RuntimeException {

    @Positive
        public JCDiagnostic getDiagnostic();
    @Positive
    }

    @Positive
    Symbol findField(Env<AttrContext> env, Type site, Name name, TypeSymbol c);

    @Positive
    public VarSymbol resolveInternalField(DiagnosticPosition pos, Env<AttrContext> env, Type site, Name name);

    @Positive
    Symbol findVar(Env<AttrContext> env, Name name);

    @Positive
    @SuppressWarnings("fallthrough")
    @Positive
    Symbol selectBest(Env<AttrContext> env, Type site, List<Type> argtypes, List<Type> typeargtypes, Symbol sym, Symbol bestSoFar, boolean allowBoxing, boolean useVarargs);

    @Positive
    Symbol mostSpecific(List<Type> argtypes, Symbol m1, Symbol m2, Env<AttrContext> env, final Type site, boolean useVarargs);

    @Positive
    List<Type> adjustArgs(List<Type> args, Symbol msym, int length, boolean allowVarargs);

    @Positive
    Symbol ambiguityError(Symbol m1, Symbol m2);

    @Positive
    Symbol findMethodInScope(Env<AttrContext> env, Type site, Name name, List<Type> argtypes, List<Type> typeargtypes, Scope sc, Symbol bestSoFar, boolean allowBoxing, boolean useVarargs, boolean abstractok);

    @Positive
    class LookupFilter implements Predicate<Symbol> {

    @Positive
        @Override
    @Positive
        public boolean test(Symbol s);
    @Positive
    }

    @Positive
    Symbol findMethod(Env<AttrContext> env, Type site, Name name, List<Type> argtypes, List<Type> typeargtypes, boolean allowBoxing, boolean useVarargs);

    @Positive
    Iterable<TypeSymbol> superclasses(final Type intype);

    @Positive
    Symbol findFun(Env<AttrContext> env, Name name, List<Type> argtypes, List<Type> typeargtypes, boolean allowBoxing, boolean useVarargs);

    @Positive
    Symbol loadClass(Env<AttrContext> env, Name name, RecoveryLoadClass recoveryLoadClass);

    @Positive
    public interface RecoveryLoadClass {

    @Positive
        Symbol loadClass(Env<AttrContext> env, Name name);
    @Positive
    }

    @Positive
    Symbol lookupPackage(Env<AttrContext> env, Name name);

    @Positive
    Symbol findImmediateMemberType(Env<AttrContext> env, Type site, Name name, TypeSymbol c);

    @Positive
    Symbol findInheritedMemberType(Env<AttrContext> env, Type site, Name name, TypeSymbol c);

    @Positive
    Symbol findMemberType(Env<AttrContext> env, Type site, Name name, TypeSymbol c);

    @Positive
    Symbol findGlobalType(Env<AttrContext> env, Scope scope, Name name, RecoveryLoadClass recoveryLoadClass);

    @Positive
    Symbol findTypeVar(Env<AttrContext> env, Name name, boolean staticOnly);

    @Positive
    Symbol findType(Env<AttrContext> env, Name name);

    @Positive
    Symbol findIdent(DiagnosticPosition pos, Env<AttrContext> env, Name name, KindSelector kind);

    @Positive
    Symbol findIdentInternal(Env<AttrContext> env, Name name, KindSelector kind);

    @Positive
    Symbol findIdentInPackage(DiagnosticPosition pos, Env<AttrContext> env, TypeSymbol pck, Name name, KindSelector kind);

    @Positive
    Symbol findIdentInPackageInternal(Env<AttrContext> env, TypeSymbol pck, Name name, KindSelector kind);

    @Positive
    Symbol findIdentInType(DiagnosticPosition pos, Env<AttrContext> env, Type site, Name name, KindSelector kind);

    @Positive
    Symbol findIdentInTypeInternal(Env<AttrContext> env, Type site, Name name, KindSelector kind);

    @Positive
    Symbol accessInternal(Symbol sym, DiagnosticPosition pos, Symbol location, Type site, Name name, boolean qualified, List<Type> argtypes, List<Type> typeargtypes, LogResolveHelper logResolveHelper);

    @Positive
    Symbol accessMethod(Symbol sym, DiagnosticPosition pos, Symbol location, Type site, Name name, boolean qualified, List<Type> argtypes, List<Type> typeargtypes);

    @Positive
    Symbol accessMethod(Symbol sym, DiagnosticPosition pos, Type site, Name name, boolean qualified, List<Type> argtypes, List<Type> typeargtypes);

    @Positive
    Symbol accessBase(Symbol sym, DiagnosticPosition pos, Symbol location, Type site, Name name, boolean qualified);

    @Positive
    Symbol accessBase(Symbol sym, DiagnosticPosition pos, Type site, Name name, boolean qualified);

    @Positive
    interface LogResolveHelper {

    @Positive
        boolean resolveDiagnosticNeeded(Type site, List<Type> argtypes, List<Type> typeargtypes);

    @Positive
        List<Type> getArgumentTypes(ResolveError errSym, Symbol accessedSym, Name name, List<Type> argtypes);
    @Positive
    }

    @Positive
    class ResolveDeferredRecoveryMap extends DeferredAttr.RecoveryDeferredTypeMap {

    @Positive
        public ResolveDeferredRecoveryMap(AttrMode mode, Symbol msym, MethodResolutionPhase step) {
    @Positive
        }

    @Positive
        @Override
    @Positive
        protected Type typeOf(DeferredType dt, Type pt);
    @Positive
    }

    @Positive
    void checkNonAbstract(DiagnosticPosition pos, Symbol sym);

    @Positive
    Symbol resolveIdent(DiagnosticPosition pos, Env<AttrContext> env, Name name, KindSelector kind);

    @Positive
    Symbol resolveMethod(DiagnosticPosition pos, Env<AttrContext> env, Name name, List<Type> argtypes, List<Type> typeargtypes);

    @Positive
    Symbol resolveQualifiedMethod(DiagnosticPosition pos, Env<AttrContext> env, Type site, Name name, List<Type> argtypes, List<Type> typeargtypes);

    @Positive
    Symbol resolveQualifiedMethod(DiagnosticPosition pos, Env<AttrContext> env, Symbol location, Type site, Name name, List<Type> argtypes, List<Type> typeargtypes);

    @Positive
    Symbol findPolymorphicSignatureInstance(Env<AttrContext> env, final Symbol spMethod, List<Type> argtypes);

    @Positive
    Symbol findPolymorphicSignatureInstance(final Symbol spMethod, Type mtype);

    @Positive
    public MethodSymbol resolveInternalMethod(DiagnosticPosition pos, Env<AttrContext> env, Type site, Name name, List<Type> argtypes, List<Type> typeargtypes);

    @Positive
    Symbol resolveConstructor(DiagnosticPosition pos, Env<AttrContext> env, Type site, List<Type> argtypes, List<Type> typeargtypes);

    @Positive
    public MethodSymbol resolveInternalConstructor(DiagnosticPosition pos, Env<AttrContext> env, Type site, List<Type> argtypes, List<Type> typeargtypes);

    @Positive
    Symbol findConstructor(DiagnosticPosition pos, Env<AttrContext> env, Type site, List<Type> argtypes, List<Type> typeargtypes, boolean allowBoxing, boolean useVarargs);

    @Positive
    Symbol resolveDiamond(DiagnosticPosition pos, Env<AttrContext> env, Type site, List<Type> argtypes, List<Type> typeargtypes);

    @Positive
    Symbol getMemberReference(DiagnosticPosition pos, Env<AttrContext> env, JCMemberReference referenceTree, Type site, Name name);

    @Positive
    ReferenceLookupHelper makeReferenceLookupHelper(JCMemberReference referenceTree, Type site, Name name, List<Type> argtypes, List<Type> typeargtypes, MethodResolutionPhase maxPhase);

    @Positive
    Pair<Symbol, ReferenceLookupHelper> resolveMemberReference(Env<AttrContext> env, JCMemberReference referenceTree, Type site, Name name, List<Type> argtypes, List<Type> typeargtypes, Type descriptor, MethodCheck methodCheck, InferenceContext inferenceContext, ReferenceChooser referenceChooser);

    @Positive
    static class ReferenceLookupResult {

    @Positive
        boolean isSuccess();

    @Positive
        boolean hasKind(StaticKind sk);

    @Positive
        boolean canIgnore();

    @Positive
        static ReferenceLookupResult error(Symbol sym);
    @Positive
    }

    @Positive
    abstract class ReferenceChooser {

    @Positive
        ReferenceLookupResult result(ReferenceLookupResult boundRes, ReferenceLookupResult unboundRes);

    @Positive
        abstract ReferenceLookupResult boundResult(ReferenceLookupResult boundRes);

    @Positive
        abstract ReferenceLookupResult unboundResult(ReferenceLookupResult boundRes, ReferenceLookupResult unboundRes);
    @Positive
    }

    @Positive
    abstract class LookupHelper {

    @Positive
        final boolean shouldStop(Symbol sym, MethodResolutionPhase phase);

    @Positive
        abstract Symbol lookup(Env<AttrContext> env, MethodResolutionPhase phase);

    @Positive
        void debug(DiagnosticPosition pos, Symbol sym);

    @Positive
        abstract Symbol access(Env<AttrContext> env, DiagnosticPosition pos, Symbol location, Symbol sym);
    @Positive
    }

    @Positive
    abstract class BasicLookupHelper extends LookupHelper {

    @Positive
        @Override
    @Positive
        final Symbol lookup(Env<AttrContext> env, MethodResolutionPhase phase);

    @Positive
        abstract Symbol doLookup(Env<AttrContext> env, MethodResolutionPhase phase);

    @Positive
        @Override
    @Positive
        Symbol access(Env<AttrContext> env, DiagnosticPosition pos, Symbol location, Symbol sym);

    @Positive
        @Override
    @Positive
        void debug(DiagnosticPosition pos, Symbol sym);
    @Positive
    }

    @Positive
    abstract class ReferenceLookupHelper extends LookupHelper {

    @Positive
        ReferenceLookupHelper unboundLookup(InferenceContext inferenceContext);

    @Positive
        abstract JCMemberReference.ReferenceKind referenceKind(Symbol sym);

    @Positive
        Symbol access(Env<AttrContext> env, DiagnosticPosition pos, Symbol location, Symbol sym);
    @Positive
    }

    @Positive
    class MethodReferenceLookupHelper extends ReferenceLookupHelper {

    @Positive
        @Override
    @Positive
        final Symbol lookup(Env<AttrContext> env, MethodResolutionPhase phase);

    @Positive
        @Override
    @Positive
        ReferenceLookupHelper unboundLookup(InferenceContext inferenceContext);

    @Positive
        @Override
    @Positive
        ReferenceKind referenceKind(Symbol sym);
    @Positive
    }

    @Positive
    class UnboundMethodReferenceLookupHelper extends MethodReferenceLookupHelper {

    @Positive
        @Override
    @Positive
        ReferenceLookupHelper unboundLookup(InferenceContext inferenceContext);

    @Positive
        @Override
    @Positive
        ReferenceKind referenceKind(Symbol sym);
    @Positive
    }

    @Positive
    class ArrayConstructorReferenceLookupHelper extends ReferenceLookupHelper {

    @Positive
        @Override
    @Positive
        protected Symbol lookup(Env<AttrContext> env, MethodResolutionPhase phase);

    @Positive
        @Override
    @Positive
        ReferenceKind referenceKind(Symbol sym);
    @Positive
    }

    @Positive
    class ConstructorReferenceLookupHelper extends ReferenceLookupHelper {

    @Positive
        @Override
    @Positive
        protected Symbol lookup(Env<AttrContext> env, MethodResolutionPhase phase);

    @Positive
        @Override
    @Positive
        ReferenceKind referenceKind(Symbol sym);
    @Positive
    }

    @Positive
    Symbol lookupMethod(Env<AttrContext> env, DiagnosticPosition pos, Symbol location, MethodCheck methodCheck, LookupHelper lookupHelper);

    @Positive
    Symbol lookupMethod(Env<AttrContext> env, DiagnosticPosition pos, Symbol location, MethodResolutionContext resolveContext, LookupHelper lookupHelper);

    @Positive
    Symbol resolveSelf(DiagnosticPosition pos, Env<AttrContext> env, TypeSymbol c, Name name);

    @Positive
    Symbol resolveSelfContaining(DiagnosticPosition pos, Env<AttrContext> env, Symbol member, boolean isSuperCall);

    @Positive
    boolean enclosingInstanceMissing(Env<AttrContext> env, Type type);

    @Positive
    Type resolveImplicitThis(DiagnosticPosition pos, Env<AttrContext> env, Type t);

    @Positive
    Type resolveImplicitThis(DiagnosticPosition pos, Env<AttrContext> env, Type t, boolean isSuperCall);

    @Positive
    public void logAccessErrorInternal(Env<AttrContext> env, JCTree tree, Type type);

    @Positive
    public Object methodArguments(List<Type> argtypes);

    @Positive
    abstract class ResolveError extends Symbol {

    @Positive
        @Override
    @Positive
        @DefinedBy(Api.LANGUAGE_MODEL)
    @Positive
        public <R, P> R accept(ElementVisitor<R, P> v, P p);

    @Positive
        @Override
    @Positive
        public String toString();

    @Positive
        @Override
    @Positive
        public boolean exists();

    @Positive
        @Override
    @Positive
        public boolean isStatic();

    @Positive
        protected Symbol access(Name name, TypeSymbol location);

    @Positive
        abstract JCDiagnostic getDiagnostic(JCDiagnostic.DiagnosticType dkind, DiagnosticPosition pos, Symbol location, Type site, Name name, List<Type> argtypes, List<Type> typeargtypes);
    @Positive
    }

    @Positive
    abstract class InvalidSymbolError extends ResolveError {

    @Positive
        @Override
    @Positive
        public boolean exists();

    @Positive
        @Override
    @Positive
        public String toString();

    @Positive
        @Override
    @Positive
        public Symbol access(Name name, TypeSymbol location);
    @Positive
    }

    @Positive
    class BadRestrictedTypeError extends ResolveError {

    @Positive
        @Override
    @Positive
        JCDiagnostic getDiagnostic(DiagnosticType dkind, DiagnosticPosition pos, Symbol location, Type site, Name name, List<Type> argtypes, List<Type> typeargtypes);
    @Positive
    }

    @Positive
    class SymbolNotFoundError extends ResolveError {

    @Positive
        @Override
    @Positive
        JCDiagnostic getDiagnostic(JCDiagnostic.DiagnosticType dkind, DiagnosticPosition pos, Symbol location, Type site, Name name, List<Type> argtypes, List<Type> typeargtypes);
    @Positive
    }

    @Positive
    class InapplicableSymbolError extends ResolveError {

    @Positive
        protected MethodResolutionContext resolveContext;

    @Positive
        protected InapplicableSymbolError(Kind kind, String debugName, MethodResolutionContext context) {
    @Positive
        }

    @Positive
        @Override
    @Positive
        public String toString();

    @Positive
        @Override
    @Positive
        public boolean exists();

    @Positive
        @Override
    @Positive
        JCDiagnostic getDiagnostic(JCDiagnostic.DiagnosticType dkind, DiagnosticPosition pos, Symbol location, Type site, Name name, List<Type> argtypes, List<Type> typeargtypes);

    @Positive
        @Override
    @Positive
        public Symbol access(Name name, TypeSymbol location);

    @Positive
        protected Pair<Symbol, JCDiagnostic> errCandidate();
    @Positive
    }

    @Positive
    class InapplicableSymbolsError extends InapplicableSymbolError {

    @Positive
        @Override
    @Positive
        JCDiagnostic getDiagnostic(JCDiagnostic.DiagnosticType dkind, DiagnosticPosition pos, Symbol location, Type site, Name name, List<Type> argtypes, List<Type> typeargtypes);

    @Positive
        @SuppressWarnings("serial")
    @Positive
        private class MostSpecificMap extends LinkedHashMap<Symbol, JCDiagnostic> {
    @Positive
        }

    @Positive
        Map<Symbol, JCDiagnostic> filterCandidates(Map<Symbol, JCDiagnostic> candidatesMap);

    @Positive
        @Override
    @Positive
        protected Pair<Symbol, JCDiagnostic> errCandidate();
    @Positive
    }

    @Positive
    class DiamondError extends InapplicableSymbolError {

    @Positive
        public DiamondError(Symbol sym, MethodResolutionContext context) {
    @Positive
        }

    @Positive
        JCDiagnostic getDetails();

    @Positive
        @Override
    @Positive
        JCDiagnostic getDiagnostic(DiagnosticType dkind, DiagnosticPosition pos, Symbol location, Type site, Name name, List<Type> argtypes, List<Type> typeargtypes);
    @Positive
    }

    @Positive
    class AccessError extends InvalidSymbolError {

    @Positive
        @Override
    @Positive
        public boolean exists();

    @Positive
        @Override
    @Positive
        JCDiagnostic getDiagnostic(JCDiagnostic.DiagnosticType dkind, DiagnosticPosition pos, Symbol location, Type site, Name name, List<Type> argtypes, List<Type> typeargtypes);
    @Positive
    }

    @Positive
    class InvisibleSymbolError extends InvalidSymbolError {

    @Positive
        @Override
    @Positive
        JCDiagnostic getDiagnostic(JCDiagnostic.DiagnosticType dkind, DiagnosticPosition pos, Symbol location, Type site, Name name, List<Type> argtypes, List<Type> typeargtypes);
    @Positive
    }

    @Positive
    JCDiagnostic inaccessiblePackageReason(Env<AttrContext> env, PackageSymbol sym);

    @Positive
    class StaticError extends InvalidSymbolError {

    @Positive
        @Override
    @Positive
        JCDiagnostic getDiagnostic(JCDiagnostic.DiagnosticType dkind, DiagnosticPosition pos, Symbol location, Type site, Name name, List<Type> argtypes, List<Type> typeargtypes);
    @Positive
    }

    @Positive
    class AmbiguityError extends ResolveError {

    @Positive
        @Override
    @Positive
        public boolean exists();

    @Positive
        AmbiguityError addAmbiguousSymbol(Symbol s);

    @Positive
        @Override
    @Positive
        JCDiagnostic getDiagnostic(JCDiagnostic.DiagnosticType dkind, DiagnosticPosition pos, Symbol location, Type site, Name name, List<Type> argtypes, List<Type> typeargtypes);

    @Positive
        Symbol mergeAbstracts(Type site);

    @Positive
        @Override
    @Positive
        protected Symbol access(Name name, TypeSymbol location);
    @Positive
    }

    @Positive
    class BadVarargsMethod extends ResolveError {

    @Positive
        @Override
    @Positive
        public Symbol baseSymbol();

    @Positive
        @Override
    @Positive
        protected Symbol access(Name name, TypeSymbol location);

    @Positive
        @Override
    @Positive
        public boolean exists();

    @Positive
        @Override
    @Positive
        JCDiagnostic getDiagnostic(DiagnosticType dkind, DiagnosticPosition pos, Symbol location, Type site, Name name, List<Type> argtypes, List<Type> typeargtypes);
    @Positive
    }

    @Positive
    class BadMethodReferenceError extends StaticError {

    @Positive
        public BadMethodReferenceError(Symbol sym, boolean unboundLookup) {
    @Positive
        }

    @Positive
        @Override
    @Positive
        JCDiagnostic getDiagnostic(DiagnosticType dkind, DiagnosticPosition pos, Symbol location, Type site, Name name, List<Type> argtypes, List<Type> typeargtypes);
    @Positive
    }

    @Positive
    class BadConstructorReferenceError extends InvalidSymbolError {

    @Positive
        public BadConstructorReferenceError(Symbol sym) {
    @Positive
        }

    @Positive
        @Override
    @Positive
        JCDiagnostic getDiagnostic(DiagnosticType dkind, DiagnosticPosition pos, Symbol location, Type site, Name name, List<Type> argtypes, List<Type> typeargtypes);
    @Positive
    }

    @Positive
    class BadClassFileError extends InvalidSymbolError {

    @Positive
        public BadClassFileError(CompletionFailure ex) {
    @Positive
        }

    @Positive
        @Override
    @Positive
        JCDiagnostic getDiagnostic(DiagnosticType dkind, DiagnosticPosition pos, Symbol location, Type site, Name name, List<Type> argtypes, List<Type> typeargtypes);
    @Positive
    }

    @Positive
    static class MethodResolutionDiagHelper {

    @Positive
        interface DiagnosticRewriter {

    @Positive
            JCDiagnostic rewriteDiagnostic(JCDiagnostic.Factory diags, DiagnosticPosition preferredPos, DiagnosticSource preferredSource, DiagnosticType preferredKind, JCDiagnostic d);
    @Positive
        }

    @Positive
        static class Template {

    @Positive
            boolean matches(Object o);
    @Positive
        }

    @Positive
        static class ArgMismatchRewriter implements DiagnosticRewriter {

    @Positive
            public ArgMismatchRewriter(int causeIndex) {
    @Positive
            }

    @Positive
            @Override
    @Positive
            public JCDiagnostic rewriteDiagnostic(JCDiagnostic.Factory diags, DiagnosticPosition preferredPos, DiagnosticSource preferredSource, DiagnosticType preferredKind, JCDiagnostic d);
    @Positive
        }

    @Positive
        static JCDiagnostic rewrite(JCDiagnostic.Factory diags, DiagnosticPosition pos, DiagnosticSource source, DiagnosticType dkind, JCDiagnostic d);
    @Positive
    }

    @Positive
    class MethodResolutionContext {

    @Positive
        void addInapplicableCandidate(Symbol sym, JCDiagnostic details);

    @Positive
        void addApplicableCandidate(Symbol sym, Type mtype);

    @Positive
        DeferredAttrContext deferredAttrContext(Symbol sym, InferenceContext inferenceContext, ResultInfo pendingResult, Warner warn);

    @Positive
        @SuppressWarnings("overrides")
    @Positive
        class Candidate {

    @Positive
            boolean isApplicable();
    @Positive
        }

    @Positive
        DeferredAttr.AttrMode attrMode();

    @Positive
        boolean internal();
    @Positive
    }
    @Positive
}

// CFWR semantic augmentation - variant 0
