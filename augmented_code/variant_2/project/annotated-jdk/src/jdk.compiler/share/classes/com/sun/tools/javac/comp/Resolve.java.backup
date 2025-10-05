/*
 * Copyright (c) 1999, 2021, Oracle and/or its affiliates. All rights reserved.
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * This code is free software; you can redistribute it and/or modify it
 * under the terms of the GNU General Public License version 2 only, as
 * published by the Free Software Foundation.  Oracle designates this
 * particular file as subject to the "Classpath" exception as provided
 * by Oracle in the LICENSE file that accompanied this code.
 *
 * This code is distributed in the hope that it will be useful, but WITHOUT
 * ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
 * FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public License
 * version 2 for more details (a copy is included in the LICENSE file that
 * accompanied this code).
 *
 * You should have received a copy of the GNU General Public License version
 * 2 along with this work; if not, write to the Free Software Foundation,
 * Inc., 51 Franklin St, Fifth Floor, Boston, MA 02110-1301 USA.
 *
 * Please contact Oracle, 500 Oracle Parkway, Redwood Shores, CA 94065 USA
 * or visit www.oracle.com if you need additional information or have any
 * questions.
 */
package com.sun.tools.javac.comp;

import org.checkerframework.dataflow.qual.Pure;
import com.sun.tools.javac.api.Formattable.LocalizedString;
import com.sun.tools.javac.code.*;
import com.sun.tools.javac.code.Scope.WriteableScope;
import com.sun.tools.javac.code.Source.Feature;
import com.sun.tools.javac.code.Symbol.*;
import com.sun.tools.javac.code.Type.*;
import com.sun.tools.javac.comp.Attr.ResultInfo;
import com.sun.tools.javac.comp.Check.CheckContext;
import com.sun.tools.javac.comp.DeferredAttr.AttrMode;
import com.sun.tools.javac.comp.DeferredAttr.DeferredAttrContext;
import com.sun.tools.javac.comp.DeferredAttr.DeferredType;
import com.sun.tools.javac.comp.Resolve.MethodResolutionContext.Candidate;
import com.sun.tools.javac.comp.Resolve.MethodResolutionDiagHelper.Template;
import com.sun.tools.javac.comp.Resolve.ReferenceLookupResult.StaticKind;
import com.sun.tools.javac.jvm.*;
import com.sun.tools.javac.main.Option;
import com.sun.tools.javac.resources.CompilerProperties.Errors;
import com.sun.tools.javac.resources.CompilerProperties.Fragments;
import com.sun.tools.javac.resources.CompilerProperties.Warnings;
import com.sun.tools.javac.tree.*;
import com.sun.tools.javac.tree.JCTree.*;
import com.sun.tools.javac.tree.JCTree.JCMemberReference.ReferenceKind;
import com.sun.tools.javac.tree.JCTree.JCPolyExpression.*;
import com.sun.tools.javac.util.*;
import com.sun.tools.javac.util.DefinedBy.Api;
import com.sun.tools.javac.util.JCDiagnostic.DiagnosticFlag;
import com.sun.tools.javac.util.JCDiagnostic.DiagnosticPosition;
import com.sun.tools.javac.util.JCDiagnostic.DiagnosticType;
import com.sun.tools.javac.util.JCDiagnostic.Warning;
import java.util.Arrays;
import java.util.Collection;
import java.util.EnumSet;
import java.util.HashSet;
import java.util.Iterator;
import java.util.LinkedHashMap;
import java.util.Map;
import java.util.Set;
import java.util.function.BiFunction;
import java.util.function.BiPredicate;
import java.util.function.Consumer;
import java.util.function.Function;
import java.util.function.Predicate;
import java.util.stream.Stream;
import java.util.stream.StreamSupport;
import javax.lang.model.element.ElementVisitor;
import static com.sun.tools.javac.code.Flags.*;
import static com.sun.tools.javac.code.Flags.BLOCK;
import static com.sun.tools.javac.code.Flags.STATIC;
import static com.sun.tools.javac.code.Kinds.*;
import static com.sun.tools.javac.code.Kinds.Kind.*;
import static com.sun.tools.javac.code.TypeTag.*;
import static com.sun.tools.javac.comp.Resolve.MethodResolutionPhase.*;
import static com.sun.tools.javac.tree.JCTree.Tag.*;
import static com.sun.tools.javac.util.Iterators.createCompoundIterator;

public class Resolve {

    protected static final Context.Key<Resolve> resolveKey;

    public final boolean allowFunctionalInterfaceMostSpecific;

    public final boolean allowModules;

    public final boolean allowRecords;

    public final boolean checkVarargsAccessAfterResolution;

    protected Resolve(Context context) {
    }

    public static Resolve instance(Context context);

    void reportVerboseResolutionDiagnostic(DiagnosticPosition dpos, Name name, Type site, List<Type> argtypes, List<Type> typeargtypes, Symbol bestSoFar);

    JCDiagnostic getVerboseApplicableCandidateDiag(int pos, Symbol sym, Type inst);

    JCDiagnostic getVerboseInapplicableCandidateDiag(int pos, Symbol sym, JCDiagnostic subDiag);

    protected static boolean isStatic(Env<AttrContext> env);

    static boolean isInitializer(Env<AttrContext> env);

    public boolean isAccessible(Env<AttrContext> env, TypeSymbol c);

    public boolean isAccessible(Env<AttrContext> env, TypeSymbol c, boolean checkInner);

    boolean isAccessible(Env<AttrContext> env, Type t);

    boolean isAccessible(Env<AttrContext> env, Type t, boolean checkInner);

    public boolean isAccessible(Env<AttrContext> env, Type site, Symbol sym);

    public boolean isAccessible(Env<AttrContext> env, Type site, Symbol sym, boolean checkInner);

    void checkAccessibleType(Env<AttrContext> env, Type t);

    Type rawInstantiate(Env<AttrContext> env, Type site, Symbol m, ResultInfo resultInfo, List<Type> argtypes, List<Type> typeargtypes, boolean allowBoxing, boolean useVarargs, Warner warn) throws Infer.InferenceException;

    Type checkMethod(Env<AttrContext> env, Type site, Symbol m, ResultInfo resultInfo, List<Type> argtypes, List<Type> typeargtypes, Warner warn);

    Type instantiate(Env<AttrContext> env, Type site, Symbol m, ResultInfo resultInfo, List<Type> argtypes, List<Type> typeargtypes, boolean allowBoxing, boolean useVarargs, Warner warn);

    interface MethodCheck {

        void argumentsAcceptable(Env<AttrContext> env, DeferredAttrContext deferredAttrContext, List<Type> argtypes, List<Type> formals, Warner warn);

        MethodCheck mostSpecificCheck(List<Type> actuals);
    }

    abstract class AbstractMethodCheck implements MethodCheck {

        @Override
        public void argumentsAcceptable(final Env<AttrContext> env, DeferredAttrContext deferredAttrContext, List<Type> argtypes, List<Type> formals, Warner warn);

        abstract void checkArg(DiagnosticPosition pos, boolean varargs, Type actual, Type formal, DeferredAttrContext deferredAttrContext, Warner warn);

        protected void reportMC(DiagnosticPosition pos, MethodCheckDiag diag, InferenceContext inferenceContext, Object... args);

        class SharedInapplicableMethodException extends InapplicableMethodException {

            SharedInapplicableMethodException setMessage(JCDiagnostic details);
        }

        public MethodCheck mostSpecificCheck(List<Type> actuals);
    }

    class MethodReferenceCheck extends AbstractMethodCheck {

        @Override
        void checkArg(DiagnosticPosition pos, boolean varargs, Type actual, Type formal, DeferredAttrContext deferredAttrContext, Warner warn);

        @Override
        public MethodCheck mostSpecificCheck(List<Type> actuals);

        @Override
        public String toString();
    }

    abstract class MethodCheckContext implements CheckContext {

        public MethodCheckContext(boolean strict, DeferredAttrContext deferredAttrContext, Warner rsWarner) {
        }

        public boolean compatible(Type found, Type req, Warner warn);

        public void report(DiagnosticPosition pos, JCDiagnostic details);

        public Warner checkWarner(DiagnosticPosition pos, Type found, Type req);

        public InferenceContext inferenceContext();

        public DeferredAttrContext deferredAttrContext();

        @Override
        public String toString();
    }

    class MethodResultInfo extends ResultInfo {

        public MethodResultInfo(Type pt, CheckContext checkContext) {
        }

        @Override
        protected Type check(DiagnosticPosition pos, Type found);

        @Override
        protected MethodResultInfo dup(Type newPt);

        @Override
        protected ResultInfo dup(CheckContext newContext);

        @Override
        protected ResultInfo dup(Type newPt, CheckContext newContext);
    }

    class MostSpecificCheck implements MethodCheck {

        @Override
        public void argumentsAcceptable(final Env<AttrContext> env, DeferredAttrContext deferredAttrContext, List<Type> formals1, List<Type> formals2, Warner warn);

        ResultInfo methodCheckResult(Type to, DeferredAttr.DeferredAttrContext deferredAttrContext, Warner rsWarner, Type actual);

        class MostSpecificCheckContext extends MethodCheckContext {

            public MostSpecificCheckContext(DeferredAttrContext deferredAttrContext, Warner rsWarner, Type actual) {
            }

            public boolean compatible(Type found, Type req, Warner warn);

            class MostSpecificFunctionReturnChecker extends DeferredAttr.PolyScanner {

                @Override
                void skip(JCTree tree);

                @Override
                public void visitConditional(JCConditional tree);

                @Override
                public void visitReference(JCMemberReference tree);

                @Override
                public void visitParens(JCParens tree);

                @Override
                public void visitLambda(JCLambda tree);
            }
        }

        public MethodCheck mostSpecificCheck(List<Type> actuals);
    }

    public static class InapplicableMethodException extends RuntimeException {

        public JCDiagnostic getDiagnostic();
    }

    Symbol findField(Env<AttrContext> env, Type site, Name name, TypeSymbol c);

    public VarSymbol resolveInternalField(DiagnosticPosition pos, Env<AttrContext> env, Type site, Name name);

    Symbol findVar(Env<AttrContext> env, Name name);

    @SuppressWarnings("fallthrough")
    Symbol selectBest(Env<AttrContext> env, Type site, List<Type> argtypes, List<Type> typeargtypes, Symbol sym, Symbol bestSoFar, boolean allowBoxing, boolean useVarargs);

    Symbol mostSpecific(List<Type> argtypes, Symbol m1, Symbol m2, Env<AttrContext> env, final Type site, boolean useVarargs);

    List<Type> adjustArgs(List<Type> args, Symbol msym, int length, boolean allowVarargs);

    Symbol ambiguityError(Symbol m1, Symbol m2);

    Symbol findMethodInScope(Env<AttrContext> env, Type site, Name name, List<Type> argtypes, List<Type> typeargtypes, Scope sc, Symbol bestSoFar, boolean allowBoxing, boolean useVarargs, boolean abstractok);

    class LookupFilter implements Predicate<Symbol> {

        @Override
        public boolean test(Symbol s);
    }

    Symbol findMethod(Env<AttrContext> env, Type site, Name name, List<Type> argtypes, List<Type> typeargtypes, boolean allowBoxing, boolean useVarargs);

    Iterable<TypeSymbol> superclasses(final Type intype);

    Symbol findFun(Env<AttrContext> env, Name name, List<Type> argtypes, List<Type> typeargtypes, boolean allowBoxing, boolean useVarargs);

    Symbol loadClass(Env<AttrContext> env, Name name, RecoveryLoadClass recoveryLoadClass);

    public interface RecoveryLoadClass {

        Symbol loadClass(Env<AttrContext> env, Name name);
    }

    Symbol lookupPackage(Env<AttrContext> env, Name name);

    Symbol findImmediateMemberType(Env<AttrContext> env, Type site, Name name, TypeSymbol c);

    Symbol findInheritedMemberType(Env<AttrContext> env, Type site, Name name, TypeSymbol c);

    Symbol findMemberType(Env<AttrContext> env, Type site, Name name, TypeSymbol c);

    Symbol findGlobalType(Env<AttrContext> env, Scope scope, Name name, RecoveryLoadClass recoveryLoadClass);

    Symbol findTypeVar(Env<AttrContext> env, Name name, boolean staticOnly);

    Symbol findType(Env<AttrContext> env, Name name);

    Symbol findIdent(DiagnosticPosition pos, Env<AttrContext> env, Name name, KindSelector kind);

    Symbol findIdentInternal(Env<AttrContext> env, Name name, KindSelector kind);

    Symbol findIdentInPackage(DiagnosticPosition pos, Env<AttrContext> env, TypeSymbol pck, Name name, KindSelector kind);

    Symbol findIdentInPackageInternal(Env<AttrContext> env, TypeSymbol pck, Name name, KindSelector kind);

    Symbol findIdentInType(DiagnosticPosition pos, Env<AttrContext> env, Type site, Name name, KindSelector kind);

    Symbol findIdentInTypeInternal(Env<AttrContext> env, Type site, Name name, KindSelector kind);

    Symbol accessInternal(Symbol sym, DiagnosticPosition pos, Symbol location, Type site, Name name, boolean qualified, List<Type> argtypes, List<Type> typeargtypes, LogResolveHelper logResolveHelper);

    Symbol accessMethod(Symbol sym, DiagnosticPosition pos, Symbol location, Type site, Name name, boolean qualified, List<Type> argtypes, List<Type> typeargtypes);

    Symbol accessMethod(Symbol sym, DiagnosticPosition pos, Type site, Name name, boolean qualified, List<Type> argtypes, List<Type> typeargtypes);

    Symbol accessBase(Symbol sym, DiagnosticPosition pos, Symbol location, Type site, Name name, boolean qualified);

    Symbol accessBase(Symbol sym, DiagnosticPosition pos, Type site, Name name, boolean qualified);

    interface LogResolveHelper {

        boolean resolveDiagnosticNeeded(Type site, List<Type> argtypes, List<Type> typeargtypes);

        List<Type> getArgumentTypes(ResolveError errSym, Symbol accessedSym, Name name, List<Type> argtypes);
    }

    class ResolveDeferredRecoveryMap extends DeferredAttr.RecoveryDeferredTypeMap {

        public ResolveDeferredRecoveryMap(AttrMode mode, Symbol msym, MethodResolutionPhase step) {
        }

        @Override
        protected Type typeOf(DeferredType dt, Type pt);
    }

    void checkNonAbstract(DiagnosticPosition pos, Symbol sym);

    Symbol resolveIdent(DiagnosticPosition pos, Env<AttrContext> env, Name name, KindSelector kind);

    Symbol resolveMethod(DiagnosticPosition pos, Env<AttrContext> env, Name name, List<Type> argtypes, List<Type> typeargtypes);

    Symbol resolveQualifiedMethod(DiagnosticPosition pos, Env<AttrContext> env, Type site, Name name, List<Type> argtypes, List<Type> typeargtypes);

    Symbol resolveQualifiedMethod(DiagnosticPosition pos, Env<AttrContext> env, Symbol location, Type site, Name name, List<Type> argtypes, List<Type> typeargtypes);

    Symbol findPolymorphicSignatureInstance(Env<AttrContext> env, final Symbol spMethod, List<Type> argtypes);

    Symbol findPolymorphicSignatureInstance(final Symbol spMethod, Type mtype);

    public MethodSymbol resolveInternalMethod(DiagnosticPosition pos, Env<AttrContext> env, Type site, Name name, List<Type> argtypes, List<Type> typeargtypes);

    Symbol resolveConstructor(DiagnosticPosition pos, Env<AttrContext> env, Type site, List<Type> argtypes, List<Type> typeargtypes);

    public MethodSymbol resolveInternalConstructor(DiagnosticPosition pos, Env<AttrContext> env, Type site, List<Type> argtypes, List<Type> typeargtypes);

    Symbol findConstructor(DiagnosticPosition pos, Env<AttrContext> env, Type site, List<Type> argtypes, List<Type> typeargtypes, boolean allowBoxing, boolean useVarargs);

    Symbol resolveDiamond(DiagnosticPosition pos, Env<AttrContext> env, Type site, List<Type> argtypes, List<Type> typeargtypes);

    Symbol getMemberReference(DiagnosticPosition pos, Env<AttrContext> env, JCMemberReference referenceTree, Type site, Name name);

    ReferenceLookupHelper makeReferenceLookupHelper(JCMemberReference referenceTree, Type site, Name name, List<Type> argtypes, List<Type> typeargtypes, MethodResolutionPhase maxPhase);

    Pair<Symbol, ReferenceLookupHelper> resolveMemberReference(Env<AttrContext> env, JCMemberReference referenceTree, Type site, Name name, List<Type> argtypes, List<Type> typeargtypes, Type descriptor, MethodCheck methodCheck, InferenceContext inferenceContext, ReferenceChooser referenceChooser);

    static class ReferenceLookupResult {

        boolean isSuccess();

        boolean hasKind(StaticKind sk);

        boolean canIgnore();

        static ReferenceLookupResult error(Symbol sym);
    }

    abstract class ReferenceChooser {

        ReferenceLookupResult result(ReferenceLookupResult boundRes, ReferenceLookupResult unboundRes);

        abstract ReferenceLookupResult boundResult(ReferenceLookupResult boundRes);

        abstract ReferenceLookupResult unboundResult(ReferenceLookupResult boundRes, ReferenceLookupResult unboundRes);
    }

    abstract class LookupHelper {

        final boolean shouldStop(Symbol sym, MethodResolutionPhase phase);

        abstract Symbol lookup(Env<AttrContext> env, MethodResolutionPhase phase);

        void debug(DiagnosticPosition pos, Symbol sym);

        abstract Symbol access(Env<AttrContext> env, DiagnosticPosition pos, Symbol location, Symbol sym);
    }

    abstract class BasicLookupHelper extends LookupHelper {

        @Override
        final Symbol lookup(Env<AttrContext> env, MethodResolutionPhase phase);

        abstract Symbol doLookup(Env<AttrContext> env, MethodResolutionPhase phase);

        @Override
        Symbol access(Env<AttrContext> env, DiagnosticPosition pos, Symbol location, Symbol sym);

        @Override
        void debug(DiagnosticPosition pos, Symbol sym);
    }

    abstract class ReferenceLookupHelper extends LookupHelper {

        ReferenceLookupHelper unboundLookup(InferenceContext inferenceContext);

        abstract JCMemberReference.ReferenceKind referenceKind(Symbol sym);

        Symbol access(Env<AttrContext> env, DiagnosticPosition pos, Symbol location, Symbol sym);
    }

    class MethodReferenceLookupHelper extends ReferenceLookupHelper {

        @Override
        final Symbol lookup(Env<AttrContext> env, MethodResolutionPhase phase);

        @Override
        ReferenceLookupHelper unboundLookup(InferenceContext inferenceContext);

        @Override
        ReferenceKind referenceKind(Symbol sym);
    }

    class UnboundMethodReferenceLookupHelper extends MethodReferenceLookupHelper {

        @Override
        ReferenceLookupHelper unboundLookup(InferenceContext inferenceContext);

        @Override
        ReferenceKind referenceKind(Symbol sym);
    }

    class ArrayConstructorReferenceLookupHelper extends ReferenceLookupHelper {

        @Override
        protected Symbol lookup(Env<AttrContext> env, MethodResolutionPhase phase);

        @Override
        ReferenceKind referenceKind(Symbol sym);
    }

    class ConstructorReferenceLookupHelper extends ReferenceLookupHelper {

        @Override
        protected Symbol lookup(Env<AttrContext> env, MethodResolutionPhase phase);

        @Override
        ReferenceKind referenceKind(Symbol sym);
    }

    Symbol lookupMethod(Env<AttrContext> env, DiagnosticPosition pos, Symbol location, MethodCheck methodCheck, LookupHelper lookupHelper);

    Symbol lookupMethod(Env<AttrContext> env, DiagnosticPosition pos, Symbol location, MethodResolutionContext resolveContext, LookupHelper lookupHelper);

    Symbol resolveSelf(DiagnosticPosition pos, Env<AttrContext> env, TypeSymbol c, Name name);

    Symbol resolveSelfContaining(DiagnosticPosition pos, Env<AttrContext> env, Symbol member, boolean isSuperCall);

    boolean enclosingInstanceMissing(Env<AttrContext> env, Type type);

    Type resolveImplicitThis(DiagnosticPosition pos, Env<AttrContext> env, Type t);

    Type resolveImplicitThis(DiagnosticPosition pos, Env<AttrContext> env, Type t, boolean isSuperCall);

    public void logAccessErrorInternal(Env<AttrContext> env, JCTree tree, Type type);

    public Object methodArguments(List<Type> argtypes);

    abstract class ResolveError extends Symbol {

        @Override
        @DefinedBy(Api.LANGUAGE_MODEL)
        public <R, P> R accept(ElementVisitor<R, P> v, P p);

        @Override
        public String toString();

        @Override
        public boolean exists();

        @Override
        public boolean isStatic();

        protected Symbol access(Name name, TypeSymbol location);

        abstract JCDiagnostic getDiagnostic(JCDiagnostic.DiagnosticType dkind, DiagnosticPosition pos, Symbol location, Type site, Name name, List<Type> argtypes, List<Type> typeargtypes);
    }

    abstract class InvalidSymbolError extends ResolveError {

        @Override
        public boolean exists();

        @Override
        public String toString();

        @Override
        public Symbol access(Name name, TypeSymbol location);
    }

    class BadRestrictedTypeError extends ResolveError {

        @Override
        JCDiagnostic getDiagnostic(DiagnosticType dkind, DiagnosticPosition pos, Symbol location, Type site, Name name, List<Type> argtypes, List<Type> typeargtypes);
    }

    class SymbolNotFoundError extends ResolveError {

        @Override
        JCDiagnostic getDiagnostic(JCDiagnostic.DiagnosticType dkind, DiagnosticPosition pos, Symbol location, Type site, Name name, List<Type> argtypes, List<Type> typeargtypes);
    }

    class InapplicableSymbolError extends ResolveError {

        protected MethodResolutionContext resolveContext;

        protected InapplicableSymbolError(Kind kind, String debugName, MethodResolutionContext context) {
        }

        @Override
        public String toString();

        @Override
        public boolean exists();

        @Override
        JCDiagnostic getDiagnostic(JCDiagnostic.DiagnosticType dkind, DiagnosticPosition pos, Symbol location, Type site, Name name, List<Type> argtypes, List<Type> typeargtypes);

        @Override
        public Symbol access(Name name, TypeSymbol location);

        protected Pair<Symbol, JCDiagnostic> errCandidate();
    }

    class InapplicableSymbolsError extends InapplicableSymbolError {

        @Override
        JCDiagnostic getDiagnostic(JCDiagnostic.DiagnosticType dkind, DiagnosticPosition pos, Symbol location, Type site, Name name, List<Type> argtypes, List<Type> typeargtypes);

        @SuppressWarnings("serial")
        private class MostSpecificMap extends LinkedHashMap<Symbol, JCDiagnostic> {
        }

        Map<Symbol, JCDiagnostic> filterCandidates(Map<Symbol, JCDiagnostic> candidatesMap);

        @Override
        protected Pair<Symbol, JCDiagnostic> errCandidate();
    }

    class DiamondError extends InapplicableSymbolError {

        public DiamondError(Symbol sym, MethodResolutionContext context) {
        }

        JCDiagnostic getDetails();

        @Override
        JCDiagnostic getDiagnostic(DiagnosticType dkind, DiagnosticPosition pos, Symbol location, Type site, Name name, List<Type> argtypes, List<Type> typeargtypes);
    }

    class AccessError extends InvalidSymbolError {

        @Override
        public boolean exists();

        @Override
        JCDiagnostic getDiagnostic(JCDiagnostic.DiagnosticType dkind, DiagnosticPosition pos, Symbol location, Type site, Name name, List<Type> argtypes, List<Type> typeargtypes);
    }

    class InvisibleSymbolError extends InvalidSymbolError {

        @Override
        JCDiagnostic getDiagnostic(JCDiagnostic.DiagnosticType dkind, DiagnosticPosition pos, Symbol location, Type site, Name name, List<Type> argtypes, List<Type> typeargtypes);
    }

    JCDiagnostic inaccessiblePackageReason(Env<AttrContext> env, PackageSymbol sym);

    class StaticError extends InvalidSymbolError {

        @Override
        JCDiagnostic getDiagnostic(JCDiagnostic.DiagnosticType dkind, DiagnosticPosition pos, Symbol location, Type site, Name name, List<Type> argtypes, List<Type> typeargtypes);
    }

    class AmbiguityError extends ResolveError {

        @Override
        public boolean exists();

        AmbiguityError addAmbiguousSymbol(Symbol s);

        @Override
        JCDiagnostic getDiagnostic(JCDiagnostic.DiagnosticType dkind, DiagnosticPosition pos, Symbol location, Type site, Name name, List<Type> argtypes, List<Type> typeargtypes);

        Symbol mergeAbstracts(Type site);

        @Override
        protected Symbol access(Name name, TypeSymbol location);
    }

    class BadVarargsMethod extends ResolveError {

        @Override
        public Symbol baseSymbol();

        @Override
        protected Symbol access(Name name, TypeSymbol location);

        @Override
        public boolean exists();

        @Override
        JCDiagnostic getDiagnostic(DiagnosticType dkind, DiagnosticPosition pos, Symbol location, Type site, Name name, List<Type> argtypes, List<Type> typeargtypes);
    }

    class BadMethodReferenceError extends StaticError {

        public BadMethodReferenceError(Symbol sym, boolean unboundLookup) {
        }

        @Override
        JCDiagnostic getDiagnostic(DiagnosticType dkind, DiagnosticPosition pos, Symbol location, Type site, Name name, List<Type> argtypes, List<Type> typeargtypes);
    }

    class BadConstructorReferenceError extends InvalidSymbolError {

        public BadConstructorReferenceError(Symbol sym) {
        }

        @Override
        JCDiagnostic getDiagnostic(DiagnosticType dkind, DiagnosticPosition pos, Symbol location, Type site, Name name, List<Type> argtypes, List<Type> typeargtypes);
    }

    class BadClassFileError extends InvalidSymbolError {

        public BadClassFileError(CompletionFailure ex) {
        }

        @Override
        JCDiagnostic getDiagnostic(DiagnosticType dkind, DiagnosticPosition pos, Symbol location, Type site, Name name, List<Type> argtypes, List<Type> typeargtypes);
    }

    static class MethodResolutionDiagHelper {

        interface DiagnosticRewriter {

            JCDiagnostic rewriteDiagnostic(JCDiagnostic.Factory diags, DiagnosticPosition preferredPos, DiagnosticSource preferredSource, DiagnosticType preferredKind, JCDiagnostic d);
        }

        static class Template {

            boolean matches(Object o);
        }

        static class ArgMismatchRewriter implements DiagnosticRewriter {

            public ArgMismatchRewriter(int causeIndex) {
            }

            @Override
            public JCDiagnostic rewriteDiagnostic(JCDiagnostic.Factory diags, DiagnosticPosition preferredPos, DiagnosticSource preferredSource, DiagnosticType preferredKind, JCDiagnostic d);
        }

        static JCDiagnostic rewrite(JCDiagnostic.Factory diags, DiagnosticPosition pos, DiagnosticSource source, DiagnosticType dkind, JCDiagnostic d);
    }

    class MethodResolutionContext {

        void addInapplicableCandidate(Symbol sym, JCDiagnostic details);

        void addApplicableCandidate(Symbol sym, Type mtype);

        DeferredAttrContext deferredAttrContext(Symbol sym, InferenceContext inferenceContext, ResultInfo pendingResult, Warner warn);

        @SuppressWarnings("overrides")
        class Candidate {

            boolean isApplicable();
        }

        DeferredAttr.AttrMode attrMode();

        boolean internal();
    }
}
