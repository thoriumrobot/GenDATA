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
package com.sun.tools.javac.code;

import org.checkerframework.dataflow.qual.Pure;
import java.lang.annotation.Annotation;
import java.util.ArrayDeque;
import java.util.Collections;
import java.util.EnumMap;
import java.util.Map;
import java.util.function.Predicate;
import javax.lang.model.type.*;
import com.sun.tools.javac.code.Symbol.*;
import com.sun.tools.javac.code.TypeMetadata.Entry;
import com.sun.tools.javac.code.Types.TypeMapping;
import com.sun.tools.javac.code.Types.UniqueType;
import com.sun.tools.javac.comp.Infer.IncorporationAction;
import com.sun.tools.javac.jvm.ClassFile;
import com.sun.tools.javac.jvm.PoolConstant;
import com.sun.tools.javac.util.*;
import com.sun.tools.javac.util.DefinedBy.Api;
import static com.sun.tools.javac.code.BoundKind.*;
import static com.sun.tools.javac.code.Flags.*;
import static com.sun.tools.javac.code.Kinds.Kind.*;
import static com.sun.tools.javac.code.TypeTag.*;

public abstract class Type extends AnnoConstruct implements TypeMirror, PoolConstant {

    protected final TypeMetadata metadata;

    public TypeMetadata getMetadata();

    public Entry getMetadataOfKind(final Entry.Kind kind);

    public static final JCNoType noType;

    public static final JCNoType recoveryType;

    public static final JCNoType stuckType;

    public static boolean moreInfo;

    public TypeSymbol tsym;

    @Override
    public int poolTag();

    @Override
    public Object poolKey(Types types);

    public boolean hasTag(TypeTag tag);

    public abstract TypeTag getTag();

    public boolean isNumeric();

    public boolean isIntegral();

    public boolean isPrimitive();

    public boolean isPrimitiveOrVoid();

    public boolean isReference();

    public boolean isNullOrReference();

    public boolean isPartial();

    public Object constValue();

    public boolean isFalse();

    public boolean isTrue();

    public Type getModelType();

    public static List<Type> getModelTypes(List<Type> ts);

    public Type getOriginalType();

    public <R, S> R accept(Type.Visitor<R, S> v, S s);

    public Type(TypeSymbol tsym, TypeMetadata metadata) {
    }

    public static abstract class StructuralTypeMapping<S> extends Types.TypeMapping<S> {

        @Override
        public Type visitClassType(ClassType t, S s);

        @Override
        public Type visitWildcardType(WildcardType wt, S s);

        @Override
        public Type visitArrayType(ArrayType t, S s);

        @Override
        public Type visitMethodType(MethodType t, S s);

        @Override
        public Type visitForAll(ForAll t, S s);
    }

    public <Z> Type map(TypeMapping<Z> mapping, Z arg);

    public <Z> Type map(TypeMapping<Z> mapping);

    public Type constType(Object constValue);

    public Type baseType();

    protected Type typeNoMetadata();

    public abstract Type cloneWithMetadata(TypeMetadata metadata);

    protected boolean needsStripping();

    public Type stripMetadataIfNeeded();

    public Type stripMetadata();

    public Type annotatedType(final List<Attribute.TypeCompound> annos);

    public boolean isAnnotated();

    @Override
    @DefinedBy(Api.LANGUAGE_MODEL)
    public List<Attribute.TypeCompound> getAnnotationMirrors();

    @Override
    @DefinedBy(Api.LANGUAGE_MODEL)
    public <A extends Annotation> A getAnnotation(Class<A> annotationType);

    @Override
    @DefinedBy(Api.LANGUAGE_MODEL)
    public <A extends Annotation> A[] getAnnotationsByType(Class<A> annotationType);

    public static List<Type> baseTypes(List<Type> ts);

    protected void appendAnnotationsString(StringBuilder sb, boolean prefix);

    protected void appendAnnotationsString(StringBuilder sb);

    @DefinedBy(Api.LANGUAGE_MODEL)
    public String toString();

    public static String toString(List<Type> ts);

    public String stringValue();

    @Override
    @DefinedBy(Api.LANGUAGE_MODEL)
    public boolean equals(Object t);

    public boolean equalsIgnoreMetadata(Type t);

    @Override
    @DefinedBy(Api.LANGUAGE_MODEL)
    public int hashCode();

    public String argtypes(boolean varargs);

    public List<Type> getTypeArguments();

    public Type getEnclosingType();

    public List<Type> getParameterTypes();

    public Type getReturnType();

    public Type getReceiverType();

    public List<Type> getThrownTypes();

    public Type getUpperBound();

    public Type getLowerBound();

    public List<Type> allparams();

    public boolean isErroneous();

    public static boolean isErroneous(List<Type> ts);

    public boolean isParameterized();

    public boolean isRaw();

    public boolean isCompound();

    public boolean isIntersection();

    public boolean isUnion();

    public boolean isInterface();

    public boolean isFinal();

    @Pure
    public boolean contains(Type t);

    @Pure
    public static boolean contains(List<Type> ts, Type t);

    @Pure
    public boolean containsAny(List<Type> ts);

    @Pure
    public static boolean containsAny(List<Type> ts1, List<Type> ts2);

    public static List<Type> filter(List<Type> ts, Predicate<Type> tf);

    public boolean isSuperBound();

    public boolean isExtendsBound();

    public boolean isUnbound();

    public Type withTypeVar(Type t);

    public MethodType asMethodType();

    public void complete();

    public TypeSymbol asElement();

    @Override
    @DefinedBy(Api.LANGUAGE_MODEL)
    public TypeKind getKind();

    @Override
    @DefinedBy(Api.LANGUAGE_MODEL)
    public <R, P> R accept(TypeVisitor<R, P> v, P p);

    public static class JCPrimitiveType extends Type implements javax.lang.model.type.PrimitiveType {

        public JCPrimitiveType(TypeTag tag, TypeSymbol tsym) {
        }

        @Override
        public JCPrimitiveType cloneWithMetadata(TypeMetadata md);

        @Override
        public boolean isNumeric();

        @Override
        public boolean isIntegral();

        @Override
        public boolean isPrimitive();

        @Override
        public TypeTag getTag();

        @Override
        public boolean isPrimitiveOrVoid();

        @Override
        public Type constType(Object constValue);

        @Override
        public String stringValue();

        @Override
        public boolean isFalse();

        @Override
        public boolean isTrue();

        @Override
        @DefinedBy(Api.LANGUAGE_MODEL)
        public <R, P> R accept(TypeVisitor<R, P> v, P p);

        @Override
        @DefinedBy(Api.LANGUAGE_MODEL)
        public TypeKind getKind();
    }

    public static class WildcardType extends Type implements javax.lang.model.type.WildcardType {

        public Type type;

        public BoundKind kind;

        public TypeVar bound;

        @Override
        public <R, S> R accept(Type.Visitor<R, S> v, S s);

        public WildcardType(Type type, BoundKind kind, TypeSymbol tsym) {
        }

        public WildcardType(Type type, BoundKind kind, TypeSymbol tsym, TypeMetadata metadata) {
        }

        public WildcardType(Type type, BoundKind kind, TypeSymbol tsym, TypeVar bound) {
        }

        public WildcardType(Type type, BoundKind kind, TypeSymbol tsym, TypeVar bound, TypeMetadata metadata) {
        }

        @Override
        public WildcardType cloneWithMetadata(TypeMetadata md);

        @Override
        public TypeTag getTag();

        @Override
        @Pure
        public boolean contains(Type t);

        public boolean isSuperBound();

        public boolean isExtendsBound();

        public boolean isUnbound();

        @Override
        public boolean isReference();

        @Override
        public boolean isNullOrReference();

        @Override
        public Type withTypeVar(Type t);

        @DefinedBy(Api.LANGUAGE_MODEL)
        public String toString();

        @DefinedBy(Api.LANGUAGE_MODEL)
        public Type getExtendsBound();

        @DefinedBy(Api.LANGUAGE_MODEL)
        public Type getSuperBound();

        @DefinedBy(Api.LANGUAGE_MODEL)
        public TypeKind getKind();

        @DefinedBy(Api.LANGUAGE_MODEL)
        public <R, P> R accept(TypeVisitor<R, P> v, P p);
    }

    public static class ClassType extends Type implements DeclaredType, LoadableConstant, javax.lang.model.type.ErrorType {

        public List<Type> typarams_field;

        public List<Type> allparams_field;

        public Type supertype_field;

        public List<Type> interfaces_field;

        public List<Type> all_interfaces_field;

        public ClassType(Type outer, List<Type> typarams, TypeSymbol tsym) {
        }

        public ClassType(Type outer, List<Type> typarams, TypeSymbol tsym, TypeMetadata metadata) {
        }

        public int poolTag();

        @Override
        public ClassType cloneWithMetadata(TypeMetadata md);

        @Override
        public TypeTag getTag();

        @Override
        public <R, S> R accept(Type.Visitor<R, S> v, S s);

        public Type constType(Object constValue);

        @DefinedBy(Api.LANGUAGE_MODEL)
        public String toString();

        @DefinedBy(Api.LANGUAGE_MODEL)
        public List<Type> getTypeArguments();

        public boolean hasErasedSupertypes();

        @DefinedBy(Api.LANGUAGE_MODEL)
        public Type getEnclosingType();

        public void setEnclosingType(Type outer);

        public List<Type> allparams();

        public boolean isErroneous();

        public boolean isParameterized();

        @Override
        public boolean isReference();

        @Override
        public boolean isNullOrReference();

        public boolean isRaw();

        @Pure
        public boolean contains(Type elem);

        public void complete();

        @DefinedBy(Api.LANGUAGE_MODEL)
        public TypeKind getKind();

        @DefinedBy(Api.LANGUAGE_MODEL)
        public <R, P> R accept(TypeVisitor<R, P> v, P p);
    }

    public static class ErasedClassType extends ClassType {

        public ErasedClassType(Type outer, TypeSymbol tsym, TypeMetadata metadata) {
        }

        @Override
        public boolean hasErasedSupertypes();
    }

    public static class UnionClassType extends ClassType implements UnionType {

        public UnionClassType(ClassType ct, List<? extends Type> alternatives) {
        }

        @Override
        public UnionClassType cloneWithMetadata(TypeMetadata md);

        public Type getLub();

        @DefinedBy(Api.LANGUAGE_MODEL)
        public java.util.List<? extends TypeMirror> getAlternatives();

        @Override
        public boolean isUnion();

        @Override
        public boolean isCompound();

        @Override
        @DefinedBy(Api.LANGUAGE_MODEL)
        public TypeKind getKind();

        @Override
        @DefinedBy(Api.LANGUAGE_MODEL)
        public <R, P> R accept(TypeVisitor<R, P> v, P p);

        public Iterable<? extends Type> getAlternativeTypes();
    }

    public static class IntersectionClassType extends ClassType implements IntersectionType {

        public boolean allInterfaces;

        public IntersectionClassType(List<Type> bounds, ClassSymbol csym, boolean allInterfaces) {
        }

        @Override
        public IntersectionClassType cloneWithMetadata(TypeMetadata md);

        @DefinedBy(Api.LANGUAGE_MODEL)
        public java.util.List<? extends TypeMirror> getBounds();

        @Override
        public boolean isCompound();

        public List<Type> getComponents();

        @Override
        public boolean isIntersection();

        public List<Type> getExplicitComponents();

        @Override
        @DefinedBy(Api.LANGUAGE_MODEL)
        public TypeKind getKind();

        @Override
        @DefinedBy(Api.LANGUAGE_MODEL)
        public <R, P> R accept(TypeVisitor<R, P> v, P p);
    }

    public static class ArrayType extends Type implements LoadableConstant, javax.lang.model.type.ArrayType {

        public Type elemtype;

        public ArrayType(Type elemtype, TypeSymbol arrayClass) {
        }

        public ArrayType(Type elemtype, TypeSymbol arrayClass, TypeMetadata metadata) {
        }

        public ArrayType(ArrayType that) {
        }

        public int poolTag();

        @Override
        public ArrayType cloneWithMetadata(TypeMetadata md);

        @Override
        public TypeTag getTag();

        public <R, S> R accept(Type.Visitor<R, S> v, S s);

        @DefinedBy(Api.LANGUAGE_MODEL)
        public String toString();

        @Override
        @DefinedBy(Api.LANGUAGE_MODEL)
        public boolean equals(Object obj);

        @DefinedBy(Api.LANGUAGE_MODEL)
        public int hashCode();

        public boolean isVarargs();

        public List<Type> allparams();

        public boolean isErroneous();

        public boolean isParameterized();

        @Override
        public boolean isReference();

        @Override
        public boolean isNullOrReference();

        public boolean isRaw();

        public ArrayType makeVarargs();

        @Pure
        public boolean contains(Type elem);

        public void complete();

        @DefinedBy(Api.LANGUAGE_MODEL)
        public Type getComponentType();

        @DefinedBy(Api.LANGUAGE_MODEL)
        public TypeKind getKind();

        @DefinedBy(Api.LANGUAGE_MODEL)
        public <R, P> R accept(TypeVisitor<R, P> v, P p);
    }

    public static class MethodType extends Type implements ExecutableType, LoadableConstant {

        public List<Type> argtypes;

        public Type restype;

        public List<Type> thrown;

        public Type recvtype;

        public MethodType(List<Type> argtypes, Type restype, List<Type> thrown, TypeSymbol methodClass) {
        }

        @Override
        public MethodType cloneWithMetadata(TypeMetadata md);

        @Override
        public TypeTag getTag();

        public <R, S> R accept(Type.Visitor<R, S> v, S s);

        @DefinedBy(Api.LANGUAGE_MODEL)
        public String toString();

        @DefinedBy(Api.LANGUAGE_MODEL)
        public List<Type> getParameterTypes();

        @DefinedBy(Api.LANGUAGE_MODEL)
        public Type getReturnType();

        @DefinedBy(Api.LANGUAGE_MODEL)
        public Type getReceiverType();

        @DefinedBy(Api.LANGUAGE_MODEL)
        public List<Type> getThrownTypes();

        public boolean isErroneous();

        @Override
        public int poolTag();

        @Pure
        public boolean contains(Type elem);

        public MethodType asMethodType();

        public void complete();

        @DefinedBy(Api.LANGUAGE_MODEL)
        public List<TypeVar> getTypeVariables();

        public TypeSymbol asElement();

        @DefinedBy(Api.LANGUAGE_MODEL)
        public TypeKind getKind();

        @DefinedBy(Api.LANGUAGE_MODEL)
        public <R, P> R accept(TypeVisitor<R, P> v, P p);
    }

    public static class PackageType extends Type implements NoType {

        @Override
        public PackageType cloneWithMetadata(TypeMetadata md);

        @Override
        public TypeTag getTag();

        @Override
        public <R, S> R accept(Type.Visitor<R, S> v, S s);

        @DefinedBy(Api.LANGUAGE_MODEL)
        public String toString();

        @DefinedBy(Api.LANGUAGE_MODEL)
        public TypeKind getKind();

        @DefinedBy(Api.LANGUAGE_MODEL)
        public <R, P> R accept(TypeVisitor<R, P> v, P p);
    }

    public static class ModuleType extends Type implements NoType {

        @Override
        public ModuleType cloneWithMetadata(TypeMetadata md);

        @Override
        public ModuleType annotatedType(List<Attribute.TypeCompound> annos);

        @Override
        public TypeTag getTag();

        @Override
        public <R, S> R accept(Type.Visitor<R, S> v, S s);

        @Override
        @DefinedBy(Api.LANGUAGE_MODEL)
        public String toString();

        @Override
        @DefinedBy(Api.LANGUAGE_MODEL)
        public TypeKind getKind();

        @Override
        @DefinedBy(Api.LANGUAGE_MODEL)
        public <R, P> R accept(TypeVisitor<R, P> v, P p);
    }

    public static class TypeVar extends Type implements TypeVariable {

        public Type lower;

        public TypeVar(Name name, Symbol owner, Type lower) {
        }

        public TypeVar(TypeSymbol tsym, Type bound, Type lower) {
        }

        public TypeVar(TypeSymbol tsym, Type bound, Type lower, TypeMetadata metadata) {
        }

        @Override
        public TypeVar cloneWithMetadata(TypeMetadata md);

        @Override
        public TypeTag getTag();

        @Override
        public <R, S> R accept(Type.Visitor<R, S> v, S s);

        @Override
        @DefinedBy(Api.LANGUAGE_MODEL)
        public Type getUpperBound();

        public void setUpperBound(Type bound);

        @Override
        @DefinedBy(Api.LANGUAGE_MODEL)
        public Type getLowerBound();

        @DefinedBy(Api.LANGUAGE_MODEL)
        public TypeKind getKind();

        public boolean isCaptured();

        @Override
        public boolean isReference();

        @Override
        public boolean isNullOrReference();

        @Override
        @DefinedBy(Api.LANGUAGE_MODEL)
        public <R, P> R accept(TypeVisitor<R, P> v, P p);
    }

    public static class CapturedType extends TypeVar {

        public WildcardType wildcard;

        public CapturedType(Name name, Symbol owner, Type upper, Type lower, WildcardType wildcard) {
        }

        public CapturedType(TypeSymbol tsym, Type bound, Type upper, Type lower, WildcardType wildcard, TypeMetadata metadata) {
        }

        @Override
        public CapturedType cloneWithMetadata(TypeMetadata md);

        @Override
        public <R, S> R accept(Type.Visitor<R, S> v, S s);

        @Override
        public boolean isCaptured();

        @Override
        @DefinedBy(Api.LANGUAGE_MODEL)
        public String toString();
    }

    public static abstract class DelegatedType extends Type {

        public Type qtype;

        public TypeTag tag;

        public DelegatedType(TypeTag tag, Type qtype) {
        }

        public DelegatedType(TypeTag tag, Type qtype, TypeMetadata metadata) {
        }

        public TypeTag getTag();

        @DefinedBy(Api.LANGUAGE_MODEL)
        public String toString();

        public List<Type> getTypeArguments();

        public Type getEnclosingType();

        public List<Type> getParameterTypes();

        public Type getReturnType();

        public Type getReceiverType();

        public List<Type> getThrownTypes();

        public List<Type> allparams();

        public Type getUpperBound();

        public boolean isErroneous();
    }

    public static class ForAll extends DelegatedType implements ExecutableType {

        public List<Type> tvars;

        public ForAll(List<Type> tvars, Type qtype) {
        }

        @Override
        public ForAll cloneWithMetadata(TypeMetadata md);

        @Override
        public <R, S> R accept(Type.Visitor<R, S> v, S s);

        @DefinedBy(Api.LANGUAGE_MODEL)
        public String toString();

        public List<Type> getTypeArguments();

        public boolean isErroneous();

        @Pure
        public boolean contains(Type elem);

        public MethodType asMethodType();

        public void complete();

        @DefinedBy(Api.LANGUAGE_MODEL)
        public List<TypeVar> getTypeVariables();

        @DefinedBy(Api.LANGUAGE_MODEL)
        public TypeKind getKind();

        @DefinedBy(Api.LANGUAGE_MODEL)
        public <R, P> R accept(TypeVisitor<R, P> v, P p);
    }

    public static class UndetVar extends DelegatedType {

        public interface UndetVarListener {

            void varBoundChanged(UndetVar uv, InferenceBound ib, Type bound, boolean update);

            default void varInstantiated(UndetVar uv);
        }

        public enum InferenceBound {

            LOWER {

                public InferenceBound complement();
            }
            , EQ {

                public InferenceBound complement();
            }
            , UPPER {

                public InferenceBound complement();
            }
            ;

            public abstract InferenceBound complement();

            public boolean lessThan(InferenceBound that);
        }

        public ArrayDeque<IncorporationAction> incorporationActions;

        protected Map<InferenceBound, List<Type>> bounds;

        public int declaredCount;

        public UndetVarListener listener;

        @Override
        public <R, S> R accept(Type.Visitor<R, S> v, S s);

        public UndetVar(TypeVar origin, UndetVarListener listener, Types types) {
        }

        @DefinedBy(Api.LANGUAGE_MODEL)
        public String toString();

        public String debugString();

        public void setThrow();

        public UndetVar dup(Types types);

        public void dupTo(UndetVar uv2, Types types);

        @Override
        public UndetVar cloneWithMetadata(TypeMetadata md);

        @Override
        public boolean isPartial();

        @Override
        public Type baseType();

        public Type getInst();

        public void setInst(Type inst);

        public List<Type> getBounds(InferenceBound... ibs);

        public List<Type> getDeclaredBounds();

        public void setBounds(InferenceBound ib, List<Type> newBounds);

        public final void addBound(InferenceBound ib, Type bound, Types types);

        public void substBounds(List<Type> from, List<Type> to, Types types);

        public final boolean isCaptured();

        public final boolean isThrows();
    }

    public static class JCNoType extends Type implements NoType {

        public JCNoType() {
        }

        @Override
        public JCNoType cloneWithMetadata(TypeMetadata md);

        @Override
        public TypeTag getTag();

        @Override
        @DefinedBy(Api.LANGUAGE_MODEL)
        public TypeKind getKind();

        @Override
        @DefinedBy(Api.LANGUAGE_MODEL)
        public <R, P> R accept(TypeVisitor<R, P> v, P p);

        @Override
        public boolean isCompound();
    }

    public static class JCVoidType extends Type implements NoType {

        public JCVoidType() {
        }

        @Override
        public JCVoidType cloneWithMetadata(TypeMetadata md);

        @Override
        public TypeTag getTag();

        @Override
        @DefinedBy(Api.LANGUAGE_MODEL)
        public TypeKind getKind();

        @Override
        public boolean isCompound();

        @Override
        @DefinedBy(Api.LANGUAGE_MODEL)
        public <R, P> R accept(TypeVisitor<R, P> v, P p);

        @Override
        public boolean isPrimitiveOrVoid();
    }

    static class BottomType extends Type implements NullType {

        public BottomType() {
        }

        @Override
        public BottomType cloneWithMetadata(TypeMetadata md);

        @Override
        public TypeTag getTag();

        @Override
        @DefinedBy(Api.LANGUAGE_MODEL)
        public TypeKind getKind();

        @Override
        public boolean isCompound();

        @Override
        @DefinedBy(Api.LANGUAGE_MODEL)
        public <R, P> R accept(TypeVisitor<R, P> v, P p);

        @Override
        public Type constType(Object value);

        @Override
        public String stringValue();

        @Override
        public boolean isNullOrReference();
    }

    public static class ErrorType extends ClassType implements javax.lang.model.type.ErrorType {

        public ErrorType(ClassSymbol c, Type originalType) {
        }

        public ErrorType(Type originalType, TypeSymbol tsym) {
        }

        @Override
        public ErrorType cloneWithMetadata(TypeMetadata md);

        @Override
        public TypeTag getTag();

        @Override
        public boolean isPartial();

        @Override
        public boolean isReference();

        @Override
        public boolean isNullOrReference();

        public ErrorType(Name name, TypeSymbol container, Type originalType) {
        }

        @Override
        public <R, S> R accept(Type.Visitor<R, S> v, S s);

        public Type constType(Object constValue);

        @DefinedBy(Api.LANGUAGE_MODEL)
        public Type getEnclosingType();

        public Type getReturnType();

        public Type asSub(Symbol sym);

        public boolean isGenType(Type t);

        public boolean isErroneous();

        public boolean isCompound();

        public boolean isInterface();

        public List<Type> allparams();

        @DefinedBy(Api.LANGUAGE_MODEL)
        public List<Type> getTypeArguments();

        @DefinedBy(Api.LANGUAGE_MODEL)
        public TypeKind getKind();

        public Type getOriginalType();

        @DefinedBy(Api.LANGUAGE_MODEL)
        public <R, P> R accept(TypeVisitor<R, P> v, P p);
    }

    public static class UnknownType extends Type {

        public UnknownType() {
        }

        @Override
        public UnknownType cloneWithMetadata(TypeMetadata md);

        @Override
        public TypeTag getTag();

        @Override
        @DefinedBy(Api.LANGUAGE_MODEL)
        public <R, P> R accept(TypeVisitor<R, P> v, P p);

        @Override
        public boolean isPartial();
    }

    public interface Visitor<R, S> {

        R visitClassType(ClassType t, S s);

        R visitWildcardType(WildcardType t, S s);

        R visitArrayType(ArrayType t, S s);

        R visitMethodType(MethodType t, S s);

        R visitPackageType(PackageType t, S s);

        R visitModuleType(ModuleType t, S s);

        R visitTypeVar(TypeVar t, S s);

        R visitCapturedType(CapturedType t, S s);

        R visitForAll(ForAll t, S s);

        R visitUndetVar(UndetVar t, S s);

        R visitErrorType(ErrorType t, S s);

        R visitType(Type t, S s);
    }
}
