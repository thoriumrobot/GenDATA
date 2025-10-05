/*
 * Copyright (c) 2003, 2021, Oracle and/or its affiliates. All rights reserved.
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
package com.sun.tools.javac.code;

import org.checkerframework.dataflow.qual.Pure;
import java.lang.ref.SoftReference;
import java.util.HashSet;
import java.util.HashMap;
import java.util.Locale;
import java.util.Map;
import java.util.Optional;
import java.util.Set;
import java.util.WeakHashMap;
import java.util.function.BiPredicate;
import java.util.function.Function;
import java.util.function.Predicate;
import java.util.stream.Collector;
import javax.tools.JavaFileObject;
import com.sun.tools.javac.code.Attribute.RetentionPolicy;
import com.sun.tools.javac.code.Lint.LintCategory;
import com.sun.tools.javac.code.Source.Feature;
import com.sun.tools.javac.code.Type.UndetVar.InferenceBound;
import com.sun.tools.javac.code.TypeMetadata.Entry.Kind;
import com.sun.tools.javac.comp.AttrContext;
import com.sun.tools.javac.comp.Check;
import com.sun.tools.javac.comp.Enter;
import com.sun.tools.javac.comp.Env;
import com.sun.tools.javac.comp.LambdaToMethod;
import com.sun.tools.javac.jvm.ClassFile;
import com.sun.tools.javac.util.*;
import static com.sun.tools.javac.code.BoundKind.*;
import static com.sun.tools.javac.code.Flags.*;
import static com.sun.tools.javac.code.Kinds.Kind.*;
import static com.sun.tools.javac.code.Scope.*;
import static com.sun.tools.javac.code.Scope.LookupKind.NON_RECURSIVE;
import static com.sun.tools.javac.code.Symbol.*;
import static com.sun.tools.javac.code.Type.*;
import static com.sun.tools.javac.code.TypeTag.*;
import static com.sun.tools.javac.jvm.ClassFile.externalize;
import com.sun.tools.javac.resources.CompilerProperties.Fragments;

public class Types {

    protected static final Context.Key<Types> typesKey;

    public final Warner noWarnings;

    public static Types instance(Context context);

    protected Types(Context context) {
    }

    public Type wildUpperBound(Type t);

    public Type cvarUpperBound(Type t);

    public Type wildLowerBound(Type t);

    public Type cvarLowerBound(Type t);

    public Type skipTypeVars(Type site, boolean capture);

    class TypeProjection extends TypeMapping<ProjectionKind> {

        public TypeProjection(List<Type> vars) {
        }

        @Override
        public Type visitClassType(ClassType t, ProjectionKind pkind);

        @Override
        public Type visitArrayType(ArrayType t, ProjectionKind s);

        @Override
        public Type visitTypeVar(TypeVar t, ProjectionKind pkind);

        class TypeArgumentProjection extends TypeMapping<ProjectionKind> {

            @Override
            public Type visitType(Type t, ProjectionKind pkind);

            @Override
            public Type visitWildcardType(WildcardType wt, ProjectionKind pkind);
        }
    }

    public Type upward(Type t, List<Type> vars);

    public List<Type> captures(Type t);

    class CaptureScanner extends SimpleVisitor<Void, Set<Type>> {

        @Override
        public Void visitType(Type t, Set<Type> types);

        @Override
        public Void visitClassType(ClassType t, Set<Type> seen);

        @Override
        public Void visitArrayType(ArrayType t, Set<Type> seen);

        @Override
        public Void visitWildcardType(WildcardType t, Set<Type> seen);

        @Override
        public Void visitTypeVar(TypeVar t, Set<Type> seen);

        @Override
        public Void visitCapturedType(CapturedType t, Set<Type> seen);
    }

    public boolean isUnbounded(Type t);

    public Type asSub(Type t, Symbol sym);

    public boolean isConvertible(Type t, Type s, Warner warn);

    public boolean isConvertible(Type t, Type s);

    public static class FunctionDescriptorLookupError extends RuntimeException {

        FunctionDescriptorLookupError setMessage(JCDiagnostic diag);

        public JCDiagnostic getDiagnostic();
    }

    class DescriptorCache {

        class FunctionDescriptor {

            public Symbol getSymbol();

            public Type getType(Type site);
        }

        class Entry {

            public Entry(FunctionDescriptor cachedDescRes, int prevMark) {
            }

            boolean matches(int mark);
        }

        FunctionDescriptor get(TypeSymbol origin) throws FunctionDescriptorLookupError;

        public FunctionDescriptor findDescriptorInternal(TypeSymbol origin, CompoundScope membersCache) throws FunctionDescriptorLookupError;

        FunctionDescriptorLookupError failure(String msg, Object... args);

        FunctionDescriptorLookupError failure(JCDiagnostic diag);
    }

    public Symbol findDescriptorSymbol(TypeSymbol origin) throws FunctionDescriptorLookupError;

    public Type findDescriptorType(Type origin) throws FunctionDescriptorLookupError;

    public boolean isFunctionalInterface(TypeSymbol tsym);

    public boolean isFunctionalInterface(Type site);

    public Type removeWildcards(Type site);

    public ClassSymbol makeFunctionalInterfaceClass(Env<AttrContext> env, Name name, Type target, long cflags);

    public List<Symbol> functionalInterfaceBridges(TypeSymbol origin);

    class DescriptorFilter implements Predicate<Symbol> {

        @Override
        public boolean test(Symbol sym);
    }

    public boolean isSubtypeUnchecked(Type t, Type s);

    public boolean isSubtypeUnchecked(Type t, Type s, Warner warn);

    public final boolean isSubtype(Type t, Type s);

    public final boolean isSubtypeNoCapture(Type t, Type s);

    public boolean isSubtype(Type t, Type s, boolean capture);

    public boolean isSubtypeUnchecked(Type t, List<Type> ts, Warner warn);

    public boolean isSubtypes(List<Type> ts, List<Type> ss);

    public boolean isSubtypesUnchecked(List<Type> ts, List<Type> ss, Warner warn);

    public boolean isSuperType(Type t, Type s);

    public boolean isSameTypes(List<Type> ts, List<Type> ss);

    public boolean isSignaturePolymorphic(MethodSymbol msym);

    public boolean isSameType(Type t, Type s);

    public boolean containedBy(Type t, Type s);

    @Pure
    boolean containsType(List<Type> ts, List<Type> ss);

    @Pure
    public boolean containsType(Type t, Type s);

    public boolean isCaptureOf(Type s, WildcardType t);

    public boolean isSameWildcard(WildcardType t, Type s);

    @Pure
    public boolean containsTypeEquivalent(List<Type> ts, List<Type> ss);

    public boolean isCastable(Type t, Type s);

    public boolean isCastable(Type t, Type s, Warner warn);

    public boolean disjointTypes(List<Type> ts, List<Type> ss);

    public boolean disjointType(Type t, Type s);

    public List<Type> cvarLowerBounds(List<Type> ts);

    public boolean notSoftSubtype(Type t, Type s);

    public boolean isReifiable(Type t);

    public boolean isArray(Type t);

    public Type elemtype(Type t);

    public Type elemtypeOrType(Type t);

    public int dimensions(Type t);

    public ArrayType makeArrayType(Type t);

    public Type asSuper(Type t, Symbol sym);

    public Type asOuterSuper(Type t, Symbol sym);

    public Type asEnclosingSuper(Type t, Symbol sym);

    public Type memberType(Type t, Symbol sym);

    public boolean isAssignable(Type t, Type s);

    public boolean isAssignable(Type t, Type s, Warner warn);

    public Type erasure(Type t);

    public List<Type> erasure(List<Type> ts);

    public Type erasureRecursive(Type t);

    public List<Type> erasureRecursive(List<Type> ts);

    public IntersectionClassType makeIntersectionType(List<Type> bounds);

    public IntersectionClassType makeIntersectionType(List<Type> bounds, boolean allInterfaces);

    public Type supertype(Type t);

    public List<Type> interfaces(Type t);

    public List<Type> directSupertypes(Type t);

    public boolean isDirectSuperInterface(TypeSymbol isym, TypeSymbol origin);

    public boolean isDerivedRaw(Type t);

    public boolean isDerivedRawInternal(Type t);

    public boolean isDerivedRaw(List<Type> ts);

    public void setBounds(TypeVar t, List<Type> bounds);

    public void setBounds(TypeVar t, List<Type> bounds, boolean allInterfaces);

    public List<Type> getBounds(TypeVar t);

    public Type classBound(Type t);

    public boolean isSubSignature(Type t, Type s);

    public boolean isSubSignature(Type t, Type s, boolean strict);

    public boolean overrideEquivalent(Type t, Type s);

    public boolean overridesObjectMethod(TypeSymbol origin, Symbol msym);

    public enum MostSpecificReturnCheck {

        BASIC {

            @Override
            public boolean test(Type mt1, Type mt2, Types types);
        }
        , RTS {

            @Override
            public boolean test(Type mt1, Type mt2, Types types);
        }
        ;

        public abstract boolean test(Type mt1, Type mt2, Types types);
    }

    public Optional<Symbol> mergeAbstracts(List<Symbol> ambiguousInOrder, Type site, boolean sigCheck);

    class ImplementationCache {

        class Entry {

            public Entry(MethodSymbol cachedImpl, Predicate<Symbol> scopeFilter, boolean checkResult, int prevMark) {
            }

            boolean matches(Predicate<Symbol> scopeFilter, boolean checkResult, int mark);
        }

        MethodSymbol get(MethodSymbol ms, TypeSymbol origin, boolean checkResult, Predicate<Symbol> implFilter);
    }

    public MethodSymbol implementation(MethodSymbol ms, TypeSymbol origin, boolean checkResult, Predicate<Symbol> implFilter);

    class MembersClosureCache extends SimpleVisitor<Scope.CompoundScope, Void> {

        class MembersScope extends CompoundScope {

            public MembersScope(CompoundScope scope) {
            }

            Predicate<Symbol> combine(Predicate<Symbol> sf);

            @Override
            public Iterable<Symbol> getSymbols(Predicate<Symbol> sf, LookupKind lookupKind);

            @Override
            public Iterable<Symbol> getSymbolsByName(Name name, Predicate<Symbol> sf, LookupKind lookupKind);

            @Override
            public int getMark();
        }

        public CompoundScope visitType(Type t, Void _unused);

        @Override
        public CompoundScope visitClassType(ClassType t, Void _unused);

        @Override
        public CompoundScope visitTypeVar(TypeVar t, Void _unused);
    }

    public CompoundScope membersClosure(Type site, boolean skipInterface);

    public MethodSymbol firstUnimplementedAbstract(ClassSymbol sym);

    public class CandidatesCache {

        public Map<Entry, List<MethodSymbol>> cache;

        class Entry {

            @Override
            public boolean equals(Object obj);

            @Override
            public int hashCode();
        }

        public List<MethodSymbol> get(Entry e);

        public void put(Entry e, List<MethodSymbol> msymbols);
    }

    public CandidatesCache candidatesCache;

    public List<MethodSymbol> interfaceCandidates(Type site, MethodSymbol ms);

    public List<MethodSymbol> prune(List<MethodSymbol> methods);

    private class MethodFilter implements Predicate<Symbol> {

        @Override
        public boolean test(Symbol s);
    }

    public boolean hasSameArgs(Type t, Type s);

    public boolean hasSameArgs(Type t, Type s, boolean strict);

    private class HasSameArgs extends TypeRelation {

        public HasSameArgs(boolean strict) {
        }

        public Boolean visitType(Type t, Type s);

        @Override
        public Boolean visitMethodType(MethodType t, Type s);

        @Override
        public Boolean visitForAll(ForAll t, Type s);

        @Override
        public Boolean visitErrorType(ErrorType t, Type s);
    }

    public List<Type> subst(List<Type> ts, List<Type> from, List<Type> to);

    public Type subst(Type t, List<Type> from, List<Type> to);

    private class Subst extends StructuralTypeMapping<Void> {

        public Subst(List<Type> from, List<Type> to) {
        }

        @Override
        public Type visitTypeVar(TypeVar t, Void ignored);

        @Override
        public Type visitClassType(ClassType t, Void ignored);

        @Override
        public Type visitWildcardType(WildcardType t, Void ignored);

        @Override
        public Type visitForAll(ForAll t, Void ignored);
    }

    public List<Type> substBounds(List<Type> tvars, List<Type> from, List<Type> to);

    public TypeVar substBound(TypeVar t, List<Type> from, List<Type> to);

    public boolean hasSameBounds(ForAll t, ForAll s);

    public List<Type> newInstances(List<Type> tvars);

    public Type createMethodTypeWithParameters(Type original, List<Type> newParams);

    public Type createMethodTypeWithThrown(Type original, List<Type> newThrown);

    public Type createMethodTypeWithReturn(Type original, Type newReturn);

    public Type createErrorType(Type originalType);

    public Type createErrorType(ClassSymbol c, Type originalType);

    public Type createErrorType(Name name, TypeSymbol container, Type originalType);

    public int rank(Type t);

    public String toString(Type t, Locale locale);

    public String toString(Symbol t, Locale locale);

    @Deprecated
    public String toString(Type t);

    public List<Type> closure(Type t);

    public Collector<Type, ClosureHolder, List<Type>> closureCollector(boolean minClosure, BiPredicate<Type, Type> shouldSkip);

    class ClosureHolder {

        void add(Type type);

        ClosureHolder merge(ClosureHolder other);

        List<Type> closure();
    }

    public List<Type> insert(List<Type> cl, Type t, BiPredicate<Type, Type> shouldSkip);

    public List<Type> insert(List<Type> cl, Type t);

    public List<Type> union(List<Type> cl1, List<Type> cl2, BiPredicate<Type, Type> shouldSkip);

    public List<Type> union(List<Type> cl1, List<Type> cl2);

    public List<Type> intersect(List<Type> cl1, List<Type> cl2);

    class TypePair {

        @Override
        public int hashCode();

        @Override
        public boolean equals(Object obj);
    }

    public Type lub(List<Type> ts);

    public Type lub(Type... ts);

    List<Type> erasedSupertypes(Type t);

    public Type glb(List<Type> ts);

    public Type glb(Type t, Type s);

    public int hashCode(Type t);

    public int hashCode(Type t, boolean strict);

    private static class HashCodeVisitor extends UnaryVisitor<Integer> {

        public Integer visitType(Type t, Void ignored);

        @Override
        public Integer visitClassType(ClassType t, Void ignored);

        @Override
        public Integer visitMethodType(MethodType t, Void ignored);

        @Override
        public Integer visitWildcardType(WildcardType t, Void ignored);

        @Override
        public Integer visitArrayType(ArrayType t, Void ignored);

        @Override
        public Integer visitTypeVar(TypeVar t, Void ignored);

        @Override
        public Integer visitUndetVar(UndetVar t, Void ignored);

        @Override
        public Integer visitErrorType(ErrorType t, Void ignored);
    }

    public boolean resultSubtype(Type t, Type s, Warner warner);

    public boolean returnTypeSubstitutable(Type r1, Type r2);

    public boolean returnTypeSubstitutable(Type r1, Type r2, Type r2res, Warner warner);

    public boolean covariantReturnType(Type t, Type s, Warner warner);

    public ClassSymbol boxedClass(Type t);

    public Type boxedTypeOrType(Type t);

    public Type unboxedType(Type t);

    public Type unboxedTypeOrType(Type t);

    public List<Type> capture(List<Type> ts);

    public Type capture(Type t);

    public List<Type> freshTypeVariables(List<Type> types);

    public void adapt(Type source, Type target, ListBuffer<Type> from, ListBuffer<Type> to) throws AdaptFailure;

    class Adapter extends SimpleVisitor<Void, Type> {

        public void adapt(Type source, Type target) throws AdaptFailure;

        @Override
        public Void visitClassType(ClassType source, Type target) throws AdaptFailure;

        @Override
        public Void visitArrayType(ArrayType source, Type target) throws AdaptFailure;

        @Override
        public Void visitWildcardType(WildcardType source, Type target) throws AdaptFailure;

        @Override
        public Void visitTypeVar(TypeVar source, Type target) throws AdaptFailure;

        @Override
        public Void visitType(Type source, Type target);
    }

    public static class AdaptFailure extends RuntimeException {
    }

    class Rewriter extends UnaryVisitor<Type> {

        @Override
        public Type visitClassType(ClassType t, Void s);

        public Type visitType(Type t, Void s);

        @Override
        public Type visitCapturedType(CapturedType t, Void s);

        @Override
        public Type visitTypeVar(TypeVar t, Void s);

        @Override
        public Type visitWildcardType(WildcardType t, Void s);

        Type B(Type t);
    }

    public static class UniqueType {

        public final Type type;

        public UniqueType(Type type, Types types) {
        }

        public int hashCode();

        public boolean equals(Object obj);

        public String toString();
    }

    public static abstract class DefaultTypeVisitor<R, S> implements Type.Visitor<R, S> {

        public final R visit(Type t, S s);

        public R visitClassType(ClassType t, S s);

        public R visitWildcardType(WildcardType t, S s);

        public R visitArrayType(ArrayType t, S s);

        public R visitMethodType(MethodType t, S s);

        public R visitPackageType(PackageType t, S s);

        public R visitModuleType(ModuleType t, S s);

        public R visitTypeVar(TypeVar t, S s);

        public R visitCapturedType(CapturedType t, S s);

        public R visitForAll(ForAll t, S s);

        public R visitUndetVar(UndetVar t, S s);

        public R visitErrorType(ErrorType t, S s);
    }

    public static abstract class DefaultSymbolVisitor<R, S> implements Symbol.Visitor<R, S> {

        public final R visit(Symbol s, S arg);

        public R visitClassSymbol(ClassSymbol s, S arg);

        public R visitMethodSymbol(MethodSymbol s, S arg);

        public R visitOperatorSymbol(OperatorSymbol s, S arg);

        public R visitPackageSymbol(PackageSymbol s, S arg);

        public R visitTypeSymbol(TypeSymbol s, S arg);

        public R visitVarSymbol(VarSymbol s, S arg);
    }

    public static abstract class SimpleVisitor<R, S> extends DefaultTypeVisitor<R, S> {

        @Override
        public R visitCapturedType(CapturedType t, S s);

        @Override
        public R visitForAll(ForAll t, S s);

        @Override
        public R visitUndetVar(UndetVar t, S s);
    }

    public static abstract class TypeRelation extends SimpleVisitor<Boolean, Type> {
    }

    public static abstract class UnaryVisitor<R> extends SimpleVisitor<R, Void> {

        public final R visit(Type t);
    }

    public static class MapVisitor<S> extends DefaultTypeVisitor<Type, S> {

        public final Type visit(Type t);

        public Type visitType(Type t, S s);
    }

    public static class TypeMapping<S> extends MapVisitor<S> implements Function<Type, Type> {

        @Override
        public Type apply(Type type);

        List<Type> visit(List<Type> ts, S s);

        @Override
        public Type visitCapturedType(CapturedType t, S s);
    }

    public RetentionPolicy getRetention(Attribute.Compound a);

    public RetentionPolicy getRetention(TypeSymbol sym);

    public static abstract class SignatureGenerator {

        public static class InvalidSignatureException extends RuntimeException {

            public Type type();
        }

        protected abstract void append(char ch);

        protected abstract void append(byte[] ba);

        protected abstract void append(Name name);

        protected void classReference(ClassSymbol c);

        protected SignatureGenerator(Types types) {
        }

        protected void reportIllegalSignature(Type t);

        public void assembleSig(Type type);

        public boolean hasTypeVar(List<Type> l);

        public void assembleClassSig(Type type);

        public void assembleParamsSig(List<Type> typarams);

        public void assembleSig(List<Type> types);
    }

    public Type constantType(LoadableConstant c);

    public void newRound();
}
