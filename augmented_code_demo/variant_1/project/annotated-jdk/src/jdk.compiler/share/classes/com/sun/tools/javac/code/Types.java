/*
    @Positive
 * Copyright (c) 2003, 2021, Oracle and/or its affiliates. All rights reserved.
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
package com.sun.tools.javac.code;

    @Positive
import org.checkerframework.dataflow.qual.Pure;
    @Positive
import java.lang.ref.SoftReference;
    @Positive
import java.util.HashSet;
    @Positive
import java.util.HashMap;
    @Positive
import java.util.Locale;
    @Positive
import java.util.Map;
    @Positive
import java.util.Optional;
    @Positive
import java.util.Set;
    @Positive
import java.util.WeakHashMap;
    @Positive
import java.util.function.BiPredicate;
    @Positive
import java.util.function.Function;
    @Positive
import java.util.function.Predicate;
    @Positive
import java.util.stream.Collector;
    @Positive
import javax.tools.JavaFileObject;
    @Positive
import com.sun.tools.javac.code.Attribute.RetentionPolicy;
    @Positive
import com.sun.tools.javac.code.Lint.LintCategory;
    @Positive
import com.sun.tools.javac.code.Source.Feature;
    @Positive
import com.sun.tools.javac.code.Type.UndetVar.InferenceBound;
    @Positive
import com.sun.tools.javac.code.TypeMetadata.Entry.Kind;
    @Positive
import com.sun.tools.javac.comp.AttrContext;
    @Positive
import com.sun.tools.javac.comp.Check;
    @Positive
import com.sun.tools.javac.comp.Enter;
    @Positive
import com.sun.tools.javac.comp.Env;
    @Positive
import com.sun.tools.javac.comp.LambdaToMethod;
    @Positive
import com.sun.tools.javac.jvm.ClassFile;
    @Positive
import com.sun.tools.javac.util.*;
    @Positive
import static com.sun.tools.javac.code.BoundKind.*;
    @Positive
import static com.sun.tools.javac.code.Flags.*;
    @Positive
import static com.sun.tools.javac.code.Kinds.Kind.*;
    @Positive
import static com.sun.tools.javac.code.Scope.*;
    @Positive
import static com.sun.tools.javac.code.Scope.LookupKind.NON_RECURSIVE;
    @Positive
import static com.sun.tools.javac.code.Symbol.*;
    @Positive
import static com.sun.tools.javac.code.Type.*;
    @Positive
import static com.sun.tools.javac.code.TypeTag.*;
    @Positive
import static com.sun.tools.javac.jvm.ClassFile.externalize;
    @Positive
import com.sun.tools.javac.resources.CompilerProperties.Fragments;

    @Positive
public class Types {

    @Positive
    protected static final Context.Key<Types> typesKey;

    @Positive
    public final Warner noWarnings;

    @Positive
    public static Types instance(Context context);

    @Positive
    protected Types(Context context) {
    @Positive
    }

    @Positive
    public Type wildUpperBound(Type t);

    @Positive
    public Type cvarUpperBound(Type t);

    @Positive
    public Type wildLowerBound(Type t);

    @Positive
    public Type cvarLowerBound(Type t);

    @Positive
    public Type skipTypeVars(Type site, boolean capture);

    @Positive
    class TypeProjection extends TypeMapping<ProjectionKind> {

    @Positive
        public TypeProjection(List<Type> vars) {
    @Positive
        }

    @Positive
        @Override
    @Positive
        public Type visitClassType(ClassType t, ProjectionKind pkind);

    @Positive
        @Override
    @Positive
        public Type visitArrayType(ArrayType t, ProjectionKind s);

    @Positive
        @Override
    @Positive
        public Type visitTypeVar(TypeVar t, ProjectionKind pkind);

    @Positive
        class TypeArgumentProjection extends TypeMapping<ProjectionKind> {

    @Positive
            @Override
    @Positive
            public Type visitType(Type t, ProjectionKind pkind);

    @Positive
            @Override
    @Positive
            public Type visitWildcardType(WildcardType wt, ProjectionKind pkind);
    @Positive
        }
    @Positive
    }

    @Positive
    public Type upward(Type t, List<Type> vars);

    @Positive
    public List<Type> captures(Type t);

    @Positive
    class CaptureScanner extends SimpleVisitor<Void, Set<Type>> {

    @Positive
        @Override
    @Positive
        public Void visitType(Type t, Set<Type> types);

    @Positive
        @Override
    @Positive
        public Void visitClassType(ClassType t, Set<Type> seen);

    @Positive
        @Override
    @Positive
        public Void visitArrayType(ArrayType t, Set<Type> seen);

    @Positive
        @Override
    @Positive
        public Void visitWildcardType(WildcardType t, Set<Type> seen);

    @Positive
        @Override
    @Positive
        public Void visitTypeVar(TypeVar t, Set<Type> seen);

    @Positive
        @Override
    @Positive
        public Void visitCapturedType(CapturedType t, Set<Type> seen);
    @Positive
    }

    @Positive
    public boolean isUnbounded(Type t);

    @Positive
    public Type asSub(Type t, Symbol sym);

    @Positive
    public boolean isConvertible(Type t, Type s, Warner warn);

    @Positive
    public boolean isConvertible(Type t, Type s);

    @Positive
    public static class FunctionDescriptorLookupError extends RuntimeException {

    @Positive
        FunctionDescriptorLookupError setMessage(JCDiagnostic diag);

    @Positive
        public JCDiagnostic getDiagnostic();
    @Positive
    }

    @Positive
    class DescriptorCache {

    @Positive
        class FunctionDescriptor {

    @Positive
            public Symbol getSymbol();

    @Positive
            public Type getType(Type site);
    @Positive
        }

    @Positive
        class Entry {

    @Positive
            public Entry(FunctionDescriptor cachedDescRes, int prevMark) {
    @Positive
            }

    @Positive
            boolean matches(int mark);
    @Positive
        }

    @Positive
        FunctionDescriptor get(TypeSymbol origin) throws FunctionDescriptorLookupError;

    @Positive
        public FunctionDescriptor findDescriptorInternal(TypeSymbol origin, CompoundScope membersCache) throws FunctionDescriptorLookupError;

    @Positive
        FunctionDescriptorLookupError failure(String msg, Object... args);

    @Positive
        FunctionDescriptorLookupError failure(JCDiagnostic diag);
    @Positive
    }

    @Positive
    public Symbol findDescriptorSymbol(TypeSymbol origin) throws FunctionDescriptorLookupError;

    @Positive
    public Type findDescriptorType(Type origin) throws FunctionDescriptorLookupError;

    @Positive
    public boolean isFunctionalInterface(TypeSymbol tsym);

    @Positive
    public boolean isFunctionalInterface(Type site);

    @Positive
    public Type removeWildcards(Type site);

    @Positive
    public ClassSymbol makeFunctionalInterfaceClass(Env<AttrContext> env, Name name, Type target, long cflags);

    @Positive
    public List<Symbol> functionalInterfaceBridges(TypeSymbol origin);

    @Positive
    class DescriptorFilter implements Predicate<Symbol> {

    @Positive
        @Override
    @Positive
        public boolean test(Symbol sym);
    @Positive
    }

    @Positive
    public boolean isSubtypeUnchecked(Type t, Type s);

    @Positive
    public boolean isSubtypeUnchecked(Type t, Type s, Warner warn);

    @Positive
    public final boolean isSubtype(Type t, Type s);

    @Positive
    public final boolean isSubtypeNoCapture(Type t, Type s);

    @Positive
    public boolean isSubtype(Type t, Type s, boolean capture);

    @Positive
    public boolean isSubtypeUnchecked(Type t, List<Type> ts, Warner warn);

    @Positive
    public boolean isSubtypes(List<Type> ts, List<Type> ss);

    @Positive
    public boolean isSubtypesUnchecked(List<Type> ts, List<Type> ss, Warner warn);

    @Positive
    public boolean isSuperType(Type t, Type s);

    @Positive
    public boolean isSameTypes(List<Type> ts, List<Type> ss);

    @Positive
    public boolean isSignaturePolymorphic(MethodSymbol msym);

    @Positive
    public boolean isSameType(Type t, Type s);

    @Positive
    public boolean containedBy(Type t, Type s);

    @Positive
    @Pure
    @Positive
    boolean containsType(List<Type> ts, List<Type> ss);

    @Positive
    @Pure
    @Positive
    public boolean containsType(Type t, Type s);

    @Positive
    public boolean isCaptureOf(Type s, WildcardType t);

    @Positive
    public boolean isSameWildcard(WildcardType t, Type s);

    @Positive
    @Pure
    @Positive
    public boolean containsTypeEquivalent(List<Type> ts, List<Type> ss);

    @Positive
    public boolean isCastable(Type t, Type s);

    @Positive
    public boolean isCastable(Type t, Type s, Warner warn);

    @Positive
    public boolean disjointTypes(List<Type> ts, List<Type> ss);

    @Positive
    public boolean disjointType(Type t, Type s);

    @Positive
    public List<Type> cvarLowerBounds(List<Type> ts);

    @Positive
    public boolean notSoftSubtype(Type t, Type s);

    @Positive
    public boolean isReifiable(Type t);

    @Positive
    public boolean isArray(Type t);

    @Positive
    public Type elemtype(Type t);

    @Positive
    public Type elemtypeOrType(Type t);

    @Positive
    public int dimensions(Type t);

    @Positive
    public ArrayType makeArrayType(Type t);

    @Positive
    public Type asSuper(Type t, Symbol sym);

    @Positive
    public Type asOuterSuper(Type t, Symbol sym);

    @Positive
    public Type asEnclosingSuper(Type t, Symbol sym);

    @Positive
    public Type memberType(Type t, Symbol sym);

    @Positive
    public boolean isAssignable(Type t, Type s);

    @Positive
    public boolean isAssignable(Type t, Type s, Warner warn);

    @Positive
    public Type erasure(Type t);

    @Positive
    public List<Type> erasure(List<Type> ts);

    @Positive
    public Type erasureRecursive(Type t);

    @Positive
    public List<Type> erasureRecursive(List<Type> ts);

    @Positive
    public IntersectionClassType makeIntersectionType(List<Type> bounds);

    @Positive
    public IntersectionClassType makeIntersectionType(List<Type> bounds, boolean allInterfaces);

    @Positive
    public Type supertype(Type t);

    @Positive
    public List<Type> interfaces(Type t);

    @Positive
    public List<Type> directSupertypes(Type t);

    @Positive
    public boolean isDirectSuperInterface(TypeSymbol isym, TypeSymbol origin);

    @Positive
    public boolean isDerivedRaw(Type t);

    @Positive
    public boolean isDerivedRawInternal(Type t);

    @Positive
    public boolean isDerivedRaw(List<Type> ts);

    @Positive
    public void setBounds(TypeVar t, List<Type> bounds);

    @Positive
    public void setBounds(TypeVar t, List<Type> bounds, boolean allInterfaces);

    @Positive
    public List<Type> getBounds(TypeVar t);

    @Positive
    public Type classBound(Type t);

    @Positive
    public boolean isSubSignature(Type t, Type s);

    @Positive
    public boolean isSubSignature(Type t, Type s, boolean strict);

    @Positive
    public boolean overrideEquivalent(Type t, Type s);

    @Positive
    public boolean overridesObjectMethod(TypeSymbol origin, Symbol msym);

    @Positive
    public enum MostSpecificReturnCheck {

    @Positive
        BASIC {

    @Positive
            @Override
    @Positive
            public boolean test(Type mt1, Type mt2, Types types);
    @Positive
        }
    @Positive
        , RTS {

    @Positive
            @Override
    @Positive
            public boolean test(Type mt1, Type mt2, Types types);
    @Positive
        }
    @Positive
        ;

    @Positive
        public abstract boolean test(Type mt1, Type mt2, Types types);
    @Positive
    }

    @Positive
    public Optional<Symbol> mergeAbstracts(List<Symbol> ambiguousInOrder, Type site, boolean sigCheck);

    @Positive
    class ImplementationCache {

    @Positive
        class Entry {

    @Positive
            public Entry(MethodSymbol cachedImpl, Predicate<Symbol> scopeFilter, boolean checkResult, int prevMark) {
    @Positive
            }

    @Positive
            boolean matches(Predicate<Symbol> scopeFilter, boolean checkResult, int mark);
    @Positive
        }

    @Positive
        MethodSymbol get(MethodSymbol ms, TypeSymbol origin, boolean checkResult, Predicate<Symbol> implFilter);
    @Positive
    }

    @Positive
    public MethodSymbol implementation(MethodSymbol ms, TypeSymbol origin, boolean checkResult, Predicate<Symbol> implFilter);

    @Positive
    class MembersClosureCache extends SimpleVisitor<Scope.CompoundScope, Void> {

    @Positive
        class MembersScope extends CompoundScope {

    @Positive
            public MembersScope(CompoundScope scope) {
    @Positive
            }

    @Positive
            Predicate<Symbol> combine(Predicate<Symbol> sf);

    @Positive
            @Override
    @Positive
            public Iterable<Symbol> getSymbols(Predicate<Symbol> sf, LookupKind lookupKind);

    @Positive
            @Override
    @Positive
            public Iterable<Symbol> getSymbolsByName(Name name, Predicate<Symbol> sf, LookupKind lookupKind);

    @Positive
            @Override
    @Positive
            public int getMark();
    @Positive
        }

    @Positive
        public CompoundScope visitType(Type t, Void _unused);

    @Positive
        @Override
    @Positive
        public CompoundScope visitClassType(ClassType t, Void _unused);

    @Positive
        @Override
    @Positive
        public CompoundScope visitTypeVar(TypeVar t, Void _unused);
    @Positive
    }

    @Positive
    public CompoundScope membersClosure(Type site, boolean skipInterface);

    @Positive
    public MethodSymbol firstUnimplementedAbstract(ClassSymbol sym);

    @Positive
    public class CandidatesCache {

    @Positive
        public Map<Entry, List<MethodSymbol>> cache;

    @Positive
        class Entry {

    @Positive
            @Override
    @Positive
            public boolean equals(Object obj);

    @Positive
            @Override
    @Positive
            public int hashCode();
    @Positive
        }

    @Positive
        public List<MethodSymbol> get(Entry e);

    @Positive
        public void put(Entry e, List<MethodSymbol> msymbols);
    @Positive
    }

    @Positive
    public CandidatesCache candidatesCache;

    @Positive
    public List<MethodSymbol> interfaceCandidates(Type site, MethodSymbol ms);

    @Positive
    public List<MethodSymbol> prune(List<MethodSymbol> methods);

    @Positive
    private class MethodFilter implements Predicate<Symbol> {

    @Positive
        @Override
    @Positive
        public boolean test(Symbol s);
    @Positive
    }

    @Positive
    public boolean hasSameArgs(Type t, Type s);

    @Positive
    public boolean hasSameArgs(Type t, Type s, boolean strict);

    @Positive
    private class HasSameArgs extends TypeRelation {

    @Positive
        public HasSameArgs(boolean strict) {
    @Positive
        }

    @Positive
        public Boolean visitType(Type t, Type s);

    @Positive
        @Override
    @Positive
        public Boolean visitMethodType(MethodType t, Type s);

    @Positive
        @Override
    @Positive
        public Boolean visitForAll(ForAll t, Type s);

    @Positive
        @Override
    @Positive
        public Boolean visitErrorType(ErrorType t, Type s);
    @Positive
    }

    @Positive
    public List<Type> subst(List<Type> ts, List<Type> from, List<Type> to);

    @Positive
    public Type subst(Type t, List<Type> from, List<Type> to);

    @Positive
    private class Subst extends StructuralTypeMapping<Void> {

    @Positive
        public Subst(List<Type> from, List<Type> to) {
    @Positive
        }

    @Positive
        @Override
    @Positive
        public Type visitTypeVar(TypeVar t, Void ignored);

    @Positive
        @Override
    @Positive
        public Type visitClassType(ClassType t, Void ignored);

    @Positive
        @Override
    @Positive
        public Type visitWildcardType(WildcardType t, Void ignored);

    @Positive
        @Override
    @Positive
        public Type visitForAll(ForAll t, Void ignored);
    @Positive
    }

    @Positive
    public List<Type> substBounds(List<Type> tvars, List<Type> from, List<Type> to);

    @Positive
    public TypeVar substBound(TypeVar t, List<Type> from, List<Type> to);

    @Positive
    public boolean hasSameBounds(ForAll t, ForAll s);

    @Positive
    public List<Type> newInstances(List<Type> tvars);

    @Positive
    public Type createMethodTypeWithParameters(Type original, List<Type> newParams);

    @Positive
    public Type createMethodTypeWithThrown(Type original, List<Type> newThrown);

    @Positive
    public Type createMethodTypeWithReturn(Type original, Type newReturn);

    @Positive
    public Type createErrorType(Type originalType);

    @Positive
    public Type createErrorType(ClassSymbol c, Type originalType);

    @Positive
    public Type createErrorType(Name name, TypeSymbol container, Type originalType);

    @Positive
    public int rank(Type t);

    @Positive
    public String toString(Type t, Locale locale);

    @Positive
    public String toString(Symbol t, Locale locale);

    @Positive
    @Deprecated
    @Positive
    public String toString(Type t);

    @Positive
    public List<Type> closure(Type t);

    @Positive
    public Collector<Type, ClosureHolder, List<Type>> closureCollector(boolean minClosure, BiPredicate<Type, Type> shouldSkip);

    @Positive
    class ClosureHolder {

    @Positive
        void add(Type type);

    @Positive
        ClosureHolder merge(ClosureHolder other);

    @Positive
        List<Type> closure();
    @Positive
    }

    @Positive
    public List<Type> insert(List<Type> cl, Type t, BiPredicate<Type, Type> shouldSkip);

    @Positive
    public List<Type> insert(List<Type> cl, Type t);

    @Positive
    public List<Type> union(List<Type> cl1, List<Type> cl2, BiPredicate<Type, Type> shouldSkip);

    @Positive
    public List<Type> union(List<Type> cl1, List<Type> cl2);

    @Positive
    public List<Type> intersect(List<Type> cl1, List<Type> cl2);

    @Positive
    class TypePair {

    @Positive
        @Override
    @Positive
        public int hashCode();

    @Positive
        @Override
    @Positive
        public boolean equals(Object obj);
    @Positive
    }

    @Positive
    public Type lub(List<Type> ts);

    @Positive
    public Type lub(Type... ts);

    @Positive
    List<Type> erasedSupertypes(Type t);

    @Positive
    public Type glb(List<Type> ts);

    @Positive
    public Type glb(Type t, Type s);

    @Positive
    public int hashCode(Type t);

    @Positive
    public int hashCode(Type t, boolean strict);

    @Positive
    private static class HashCodeVisitor extends UnaryVisitor<Integer> {

    @Positive
        public Integer visitType(Type t, Void ignored);

    @Positive
        @Override
    @Positive
        public Integer visitClassType(ClassType t, Void ignored);

    @Positive
        @Override
    @Positive
        public Integer visitMethodType(MethodType t, Void ignored);

    @Positive
        @Override
    @Positive
        public Integer visitWildcardType(WildcardType t, Void ignored);

    @Positive
        @Override
    @Positive
        public Integer visitArrayType(ArrayType t, Void ignored);

    @Positive
        @Override
    @Positive
        public Integer visitTypeVar(TypeVar t, Void ignored);

    @Positive
        @Override
    @Positive
        public Integer visitUndetVar(UndetVar t, Void ignored);

    @Positive
        @Override
    @Positive
        public Integer visitErrorType(ErrorType t, Void ignored);
    @Positive
    }

    @Positive
    public boolean resultSubtype(Type t, Type s, Warner warner);

    @Positive
    public boolean returnTypeSubstitutable(Type r1, Type r2);

    @Positive
    public boolean returnTypeSubstitutable(Type r1, Type r2, Type r2res, Warner warner);

    @Positive
    public boolean covariantReturnType(Type t, Type s, Warner warner);

    @Positive
    public ClassSymbol boxedClass(Type t);

    @Positive
    public Type boxedTypeOrType(Type t);

    @Positive
    public Type unboxedType(Type t);

    @Positive
    public Type unboxedTypeOrType(Type t);

    @Positive
    public List<Type> capture(List<Type> ts);

    @Positive
    public Type capture(Type t);

    @Positive
    public List<Type> freshTypeVariables(List<Type> types);

    @Positive
    public void adapt(Type source, Type target, ListBuffer<Type> from, ListBuffer<Type> to) throws AdaptFailure;

    @Positive
    class Adapter extends SimpleVisitor<Void, Type> {

    @Positive
        public void adapt(Type source, Type target) throws AdaptFailure;

    @Positive
        @Override
    @Positive
        public Void visitClassType(ClassType source, Type target) throws AdaptFailure;

    @Positive
        @Override
    @Positive
        public Void visitArrayType(ArrayType source, Type target) throws AdaptFailure;

    @Positive
        @Override
    @Positive
        public Void visitWildcardType(WildcardType source, Type target) throws AdaptFailure;

    @Positive
        @Override
    @Positive
        public Void visitTypeVar(TypeVar source, Type target) throws AdaptFailure;

    @Positive
        @Override
    @Positive
        public Void visitType(Type source, Type target);
    @Positive
    }

    @Positive
    public static class AdaptFailure extends RuntimeException {
    @Positive
    }

    @Positive
    class Rewriter extends UnaryVisitor<Type> {

    @Positive
        @Override
    @Positive
        public Type visitClassType(ClassType t, Void s);

    @Positive
        public Type visitType(Type t, Void s);

    @Positive
        @Override
    @Positive
        public Type visitCapturedType(CapturedType t, Void s);

    @Positive
        @Override
    @Positive
        public Type visitTypeVar(TypeVar t, Void s);

    @Positive
        @Override
    @Positive
        public Type visitWildcardType(WildcardType t, Void s);

    @Positive
        Type B(Type t);
    @Positive
    }

    @Positive
    public static class UniqueType {

    @Positive
        public final Type type;

    @Positive
        public UniqueType(Type type, Types types) {
    @Positive
        }

    @Positive
        public int hashCode();

    @Positive
        public boolean equals(Object obj);

    @Positive
        public String toString();
    @Positive
    }

    @Positive
    public static abstract class DefaultTypeVisitor<R, S> implements Type.Visitor<R, S> {

    @Positive
        public final R visit(Type t, S s);

    @Positive
        public R visitClassType(ClassType t, S s);

    @Positive
        public R visitWildcardType(WildcardType t, S s);

    @Positive
        public R visitArrayType(ArrayType t, S s);

    @Positive
        public R visitMethodType(MethodType t, S s);

    @Positive
        public R visitPackageType(PackageType t, S s);

    @Positive
        public R visitModuleType(ModuleType t, S s);

    @Positive
        public R visitTypeVar(TypeVar t, S s);

    @Positive
        public R visitCapturedType(CapturedType t, S s);

    @Positive
        public R visitForAll(ForAll t, S s);

    @Positive
        public R visitUndetVar(UndetVar t, S s);

    @Positive
        public R visitErrorType(ErrorType t, S s);
    @Positive
    }

    @Positive
    public static abstract class DefaultSymbolVisitor<R, S> implements Symbol.Visitor<R, S> {

    @Positive
        public final R visit(Symbol s, S arg);

    @Positive
        public R visitClassSymbol(ClassSymbol s, S arg);

    @Positive
        public R visitMethodSymbol(MethodSymbol s, S arg);

    @Positive
        public R visitOperatorSymbol(OperatorSymbol s, S arg);

    @Positive
        public R visitPackageSymbol(PackageSymbol s, S arg);

    @Positive
        public R visitTypeSymbol(TypeSymbol s, S arg);

    @Positive
        public R visitVarSymbol(VarSymbol s, S arg);
    @Positive
    }

    @Positive
    public static abstract class SimpleVisitor<R, S> extends DefaultTypeVisitor<R, S> {

    @Positive
        @Override
    @Positive
        public R visitCapturedType(CapturedType t, S s);

    @Positive
        @Override
    @Positive
        public R visitForAll(ForAll t, S s);

    @Positive
        @Override
    @Positive
        public R visitUndetVar(UndetVar t, S s);
    @Positive
    }

    @Positive
    public static abstract class TypeRelation extends SimpleVisitor<Boolean, Type> {
    @Positive
    }

    @Positive
    public static abstract class UnaryVisitor<R> extends SimpleVisitor<R, Void> {

    @Positive
        public final R visit(Type t);
    @Positive
    }

    @Positive
    public static class MapVisitor<S> extends DefaultTypeVisitor<Type, S> {

    @Positive
        public final Type visit(Type t);

    @Positive
        public Type visitType(Type t, S s);
    @Positive
    }

    @Positive
    public static class TypeMapping<S> extends MapVisitor<S> implements Function<Type, Type> {

    @Positive
        @Override
    @Positive
        public Type apply(Type type);

    @Positive
        List<Type> visit(List<Type> ts, S s);

    @Positive
        @Override
    @Positive
        public Type visitCapturedType(CapturedType t, S s);
    @Positive
    }

    @Positive
    public RetentionPolicy getRetention(Attribute.Compound a);

    @Positive
    public RetentionPolicy getRetention(TypeSymbol sym);

    @Positive
    public static abstract class SignatureGenerator {

    @Positive
        public static class InvalidSignatureException extends RuntimeException {

    @Positive
            public Type type();
    @Positive
        }

    @Positive
        protected abstract void append(char ch);

    @Positive
        protected abstract void append(byte[] ba);

    @Positive
        protected abstract void append(Name name);

    @Positive
        protected void classReference(ClassSymbol c);

    @Positive
        protected SignatureGenerator(Types types) {
    @Positive
        }

    @Positive
        protected void reportIllegalSignature(Type t);

    @Positive
        public void assembleSig(Type type);

    @Positive
        public boolean hasTypeVar(List<Type> l);

    @Positive
        public void assembleClassSig(Type type);

    @Positive
        public void assembleParamsSig(List<Type> typarams);

    @Positive
        public void assembleSig(List<Type> types);
    @Positive
    }

    @Positive
    public Type constantType(LoadableConstant c);

    @Positive
    public void newRound();
    @Positive
}

// CFWR semantic augmentation - variant 1
