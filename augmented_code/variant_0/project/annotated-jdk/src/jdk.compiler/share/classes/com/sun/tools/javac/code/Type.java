/*
 * CFWR semantic augmentation: applied semantic-preserving transformations.
 */
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
package com.sun.tools.javac.code;

    @Positive
import org.checkerframework.dataflow.qual.Pure;
    @Positive
import java.lang.annotation.Annotation;
    @Positive
import java.util.ArrayDeque;
    @Positive
import java.util.Collections;
    @Positive
import java.util.EnumMap;
    @Positive
import java.util.Map;
    @Positive
import java.util.function.Predicate;
    @Positive
import javax.lang.model.type.*;
    @Positive
import com.sun.tools.javac.code.Symbol.*;
    @Positive
import com.sun.tools.javac.code.TypeMetadata.Entry;
    @Positive
import com.sun.tools.javac.code.Types.TypeMapping;
    @Positive
import com.sun.tools.javac.code.Types.UniqueType;
    @Positive
import com.sun.tools.javac.comp.Infer.IncorporationAction;
    @Positive
import com.sun.tools.javac.jvm.ClassFile;
    @Positive
import com.sun.tools.javac.jvm.PoolConstant;
    @Positive
import com.sun.tools.javac.util.*;
    @Positive
import com.sun.tools.javac.util.DefinedBy.Api;
    @Positive
import static com.sun.tools.javac.code.BoundKind.*;
    @Positive
import static com.sun.tools.javac.code.Flags.*;
    @Positive
import static com.sun.tools.javac.code.Kinds.Kind.*;
    @Positive
import static com.sun.tools.javac.code.TypeTag.*;

    @Positive
public abstract class Type extends AnnoConstruct implements TypeMirror, PoolConstant {

    @Positive
    protected final TypeMetadata metadata;

    @Positive
    public TypeMetadata getMetadata();

    @Positive
    public Entry getMetadataOfKind(final Entry.Kind kind);

    @Positive
    public static final JCNoType noType;

    @Positive
    public static final JCNoType recoveryType;

    @Positive
    public static final JCNoType stuckType;

    @Positive
    public static boolean moreInfo;

    @Positive
    public TypeSymbol tsym;

    @Positive
    @Override
    @Positive
    public int poolTag();

    @Positive
    @Override
    @Positive
    public Object poolKey(Types types);

    @Positive
    public boolean hasTag(TypeTag tag);

    @Positive
    public abstract TypeTag getTag();

    @Positive
    public boolean isNumeric();

    @Positive
    public boolean isIntegral();

    @Positive
    public boolean isPrimitive();

    @Positive
    public boolean isPrimitiveOrVoid();

    @Positive
    public boolean isReference();

    @Positive
    public boolean isNullOrReference();

    @Positive
    public boolean isPartial();

    @Positive
    public Object constValue();

    @Positive
    public boolean isFalse();

    @Positive
    public boolean isTrue();

    @Positive
    public Type getModelType();

    @Positive
    public static List<Type> getModelTypes(List<Type> ts);

    @Positive
    public Type getOriginalType();

    @Positive
    public <R, S> R accept(Type.Visitor<R, S> v, S s);

    @Positive
    public Type(TypeSymbol tsym, TypeMetadata metadata) {
    @Positive
    }

    @Positive
    public static abstract class StructuralTypeMapping<S> extends Types.TypeMapping<S> {

    @Positive
        @Override
    @Positive
        public Type visitClassType(ClassType t, S s);

    @Positive
        @Override
    @Positive
        public Type visitWildcardType(WildcardType wt, S s);

    @Positive
        @Override
    @Positive
        public Type visitArrayType(ArrayType t, S s);

    @Positive
        @Override
    @Positive
        public Type visitMethodType(MethodType t, S s);

    @Positive
        @Override
    @Positive
        public Type visitForAll(ForAll t, S s);
    @Positive
    }

    @Positive
    public <Z> Type map(TypeMapping<Z> mapping, Z arg);

    @Positive
    public <Z> Type map(TypeMapping<Z> mapping);

    @Positive
    public Type constType(Object constValue);

    @Positive
    public Type baseType();

    @Positive
    protected Type typeNoMetadata();

    @Positive
    public abstract Type cloneWithMetadata(TypeMetadata metadata);

    @Positive
    protected boolean needsStripping();

    @Positive
    public Type stripMetadataIfNeeded();

    @Positive
    public Type stripMetadata();

    @Positive
    public Type annotatedType(final List<Attribute.TypeCompound> annos);

    @Positive
    public boolean isAnnotated();

    @Positive
    @Override
    @Positive
    @DefinedBy(Api.LANGUAGE_MODEL)
    @Positive
    public List<Attribute.TypeCompound> getAnnotationMirrors();

    @Positive
    @Override
    @Positive
    @DefinedBy(Api.LANGUAGE_MODEL)
    @Positive
    public <A extends Annotation> A getAnnotation(Class<A> annotationType);

    @Positive
    @Override
    @Positive
    @DefinedBy(Api.LANGUAGE_MODEL)
    @Positive
    public <A extends Annotation> A[] getAnnotationsByType(Class<A> annotationType);

    @Positive
    public static List<Type> baseTypes(List<Type> ts);

    @Positive
    protected void appendAnnotationsString(StringBuilder sb, boolean prefix);

    @Positive
    protected void appendAnnotationsString(StringBuilder sb);

    @Positive
    @DefinedBy(Api.LANGUAGE_MODEL)
    @Positive
    public String toString();

    @Positive
    public static String toString(List<Type> ts);

    @Positive
    public String stringValue();

    @Positive
    @Override
    @Positive
    @DefinedBy(Api.LANGUAGE_MODEL)
    @Positive
    public boolean equals(Object t);

    @Positive
    public boolean equalsIgnoreMetadata(Type t);

    @Positive
    @Override
    @Positive
    @DefinedBy(Api.LANGUAGE_MODEL)
    @Positive
    public int hashCode();

    @Positive
    public String argtypes(boolean varargs);

    @Positive
    public List<Type> getTypeArguments();

    @Positive
    public Type getEnclosingType();

    @Positive
    public List<Type> getParameterTypes();

    @Positive
    public Type getReturnType();

    @Positive
    public Type getReceiverType();

    @Positive
    public List<Type> getThrownTypes();

    @Positive
    public Type getUpperBound();

    @Positive
    public Type getLowerBound();

    @Positive
    public List<Type> allparams();

    @Positive
    public boolean isErroneous();

    @Positive
    public static boolean isErroneous(List<Type> ts);

    @Positive
    public boolean isParameterized();

    @Positive
    public boolean isRaw();

    @Positive
    public boolean isCompound();

    @Positive
    public boolean isIntersection();

    @Positive
    public boolean isUnion();

    @Positive
    public boolean isInterface();

    @Positive
    public boolean isFinal();

    @Positive
    @Pure
    @Positive
    public boolean contains(Type t);

    @Positive
    @Pure
    @Positive
    public static boolean contains(List<Type> ts, Type t);

    @Positive
    @Pure
    @Positive
    public boolean containsAny(List<Type> ts);

    @Positive
    @Pure
    @Positive
    public static boolean containsAny(List<Type> ts1, List<Type> ts2);

    @Positive
    public static List<Type> filter(List<Type> ts, Predicate<Type> tf);

    @Positive
    public boolean isSuperBound();

    @Positive
    public boolean isExtendsBound();

    @Positive
    public boolean isUnbound();

    @Positive
    public Type withTypeVar(Type t);

    @Positive
    public MethodType asMethodType();

    @Positive
    public void complete();

    @Positive
    public TypeSymbol asElement();

    @Positive
    @Override
    @Positive
    @DefinedBy(Api.LANGUAGE_MODEL)
    @Positive
    public TypeKind getKind();

    @Positive
    @Override
    @Positive
    @DefinedBy(Api.LANGUAGE_MODEL)
    @Positive
    public <R, P> R accept(TypeVisitor<R, P> v, P p);

    @Positive
    public static class JCPrimitiveType extends Type implements javax.lang.model.type.PrimitiveType {

    @Positive
        public JCPrimitiveType(TypeTag tag, TypeSymbol tsym) {
    @Positive
        }

    @Positive
        @Override
    @Positive
        public JCPrimitiveType cloneWithMetadata(TypeMetadata md);

    @Positive
        @Override
    @Positive
        public boolean isNumeric();

    @Positive
        @Override
    @Positive
        public boolean isIntegral();

    @Positive
        @Override
    @Positive
        public boolean isPrimitive();

    @Positive
        @Override
    @Positive
        public TypeTag getTag();

    @Positive
        @Override
    @Positive
        public boolean isPrimitiveOrVoid();

    @Positive
        @Override
    @Positive
        public Type constType(Object constValue);

    @Positive
        @Override
    @Positive
        public String stringValue();

    @Positive
        @Override
    @Positive
        public boolean isFalse();

    @Positive
        @Override
    @Positive
        public boolean isTrue();

    @Positive
        @Override
    @Positive
        @DefinedBy(Api.LANGUAGE_MODEL)
    @Positive
        public <R, P> R accept(TypeVisitor<R, P> v, P p);

    @Positive
        @Override
    @Positive
        @DefinedBy(Api.LANGUAGE_MODEL)
    @Positive
        public TypeKind getKind();
    @Positive
    }

    @Positive
    public static class WildcardType extends Type implements javax.lang.model.type.WildcardType {

    @Positive
        public Type type;

    @Positive
        public BoundKind kind;

    @Positive
        public TypeVar bound;

    @Positive
        @Override
    @Positive
        public <R, S> R accept(Type.Visitor<R, S> v, S s);

    @Positive
        public WildcardType(Type type, BoundKind kind, TypeSymbol tsym) {
    @Positive
        }

    @Positive
        public WildcardType(Type type, BoundKind kind, TypeSymbol tsym, TypeMetadata metadata) {
    @Positive
        }

    @Positive
        public WildcardType(Type type, BoundKind kind, TypeSymbol tsym, TypeVar bound) {
    @Positive
        }

    @Positive
        public WildcardType(Type type, BoundKind kind, TypeSymbol tsym, TypeVar bound, TypeMetadata metadata) {
    @Positive
        }

    @Positive
        @Override
    @Positive
        public WildcardType cloneWithMetadata(TypeMetadata md);

    @Positive
        @Override
    @Positive
        public TypeTag getTag();

    @Positive
        @Override
    @Positive
        @Pure
    @Positive
        public boolean contains(Type t);

    @Positive
        public boolean isSuperBound();

    @Positive
        public boolean isExtendsBound();

    @Positive
        public boolean isUnbound();

    @Positive
        @Override
    @Positive
        public boolean isReference();

    @Positive
        @Override
    @Positive
        public boolean isNullOrReference();

    @Positive
        @Override
    @Positive
        public Type withTypeVar(Type t);

    @Positive
        @DefinedBy(Api.LANGUAGE_MODEL)
    @Positive
        public String toString();

    @Positive
        @DefinedBy(Api.LANGUAGE_MODEL)
    @Positive
        public Type getExtendsBound();

    @Positive
        @DefinedBy(Api.LANGUAGE_MODEL)
    @Positive
        public Type getSuperBound();

    @Positive
        @DefinedBy(Api.LANGUAGE_MODEL)
    @Positive
        public TypeKind getKind();

    @Positive
        @DefinedBy(Api.LANGUAGE_MODEL)
    @Positive
        public <R, P> R accept(TypeVisitor<R, P> v, P p);
    @Positive
    }

    @Positive
    public static class ClassType extends Type implements DeclaredType, LoadableConstant, javax.lang.model.type.ErrorType {

    @Positive
        public List<Type> typarams_field;

    @Positive
        public List<Type> allparams_field;

    @Positive
        public Type supertype_field;

    @Positive
        public List<Type> interfaces_field;

    @Positive
        public List<Type> all_interfaces_field;

    @Positive
        public ClassType(Type outer, List<Type> typarams, TypeSymbol tsym) {
    @Positive
        }

    @Positive
        public ClassType(Type outer, List<Type> typarams, TypeSymbol tsym, TypeMetadata metadata) {
    @Positive
        }

    @Positive
        public int poolTag();

    @Positive
        @Override
    @Positive
        public ClassType cloneWithMetadata(TypeMetadata md);

    @Positive
        @Override
    @Positive
        public TypeTag getTag();

    @Positive
        @Override
    @Positive
        public <R, S> R accept(Type.Visitor<R, S> v, S s);

    @Positive
        public Type constType(Object constValue);

    @Positive
        @DefinedBy(Api.LANGUAGE_MODEL)
    @Positive
        public String toString();

    @Positive
        @DefinedBy(Api.LANGUAGE_MODEL)
    @Positive
        public List<Type> getTypeArguments();

    @Positive
        public boolean hasErasedSupertypes();

    @Positive
        @DefinedBy(Api.LANGUAGE_MODEL)
    @Positive
        public Type getEnclosingType();

    @Positive
        public void setEnclosingType(Type outer);

    @Positive
        public List<Type> allparams();

    @Positive
        public boolean isErroneous();

    @Positive
        public boolean isParameterized();

    @Positive
        @Override
    @Positive
        public boolean isReference();

    @Positive
        @Override
    @Positive
        public boolean isNullOrReference();

    @Positive
        public boolean isRaw();

    @Positive
        @Pure
    @Positive
        public boolean contains(Type elem);

    @Positive
        public void complete();

    @Positive
        @DefinedBy(Api.LANGUAGE_MODEL)
    @Positive
        public TypeKind getKind();

    @Positive
        @DefinedBy(Api.LANGUAGE_MODEL)
    @Positive
        public <R, P> R accept(TypeVisitor<R, P> v, P p);
    @Positive
    }

    @Positive
    public static class ErasedClassType extends ClassType {

    @Positive
        public ErasedClassType(Type outer, TypeSymbol tsym, TypeMetadata metadata) {
    @Positive
        }

    @Positive
        @Override
    @Positive
        public boolean hasErasedSupertypes();
    @Positive
    }

    @Positive
    public static class UnionClassType extends ClassType implements UnionType {

    @Positive
        public UnionClassType(ClassType ct, List<? extends Type> alternatives) {
    @Positive
        }

    @Positive
        @Override
    @Positive
        public UnionClassType cloneWithMetadata(TypeMetadata md);

    @Positive
        public Type getLub();

    @Positive
        @DefinedBy(Api.LANGUAGE_MODEL)
    @Positive
        public java.util.List<? extends TypeMirror> getAlternatives();

    @Positive
        @Override
    @Positive
        public boolean isUnion();

    @Positive
        @Override
    @Positive
        public boolean isCompound();

    @Positive
        @Override
    @Positive
        @DefinedBy(Api.LANGUAGE_MODEL)
    @Positive
        public TypeKind getKind();

    @Positive
        @Override
    @Positive
        @DefinedBy(Api.LANGUAGE_MODEL)
    @Positive
        public <R, P> R accept(TypeVisitor<R, P> v, P p);

    @Positive
        public Iterable<? extends Type> getAlternativeTypes();
    @Positive
    }

    @Positive
    public static class IntersectionClassType extends ClassType implements IntersectionType {

    @Positive
        public boolean allInterfaces;

    @Positive
        public IntersectionClassType(List<Type> bounds, ClassSymbol csym, boolean allInterfaces) {
    @Positive
        }

    @Positive
        @Override
    @Positive
        public IntersectionClassType cloneWithMetadata(TypeMetadata md);

    @Positive
        @DefinedBy(Api.LANGUAGE_MODEL)
    @Positive
        public java.util.List<? extends TypeMirror> getBounds();

    @Positive
        @Override
    @Positive
        public boolean isCompound();

    @Positive
        public List<Type> getComponents();

    @Positive
        @Override
    @Positive
        public boolean isIntersection();

    @Positive
        public List<Type> getExplicitComponents();

    @Positive
        @Override
    @Positive
        @DefinedBy(Api.LANGUAGE_MODEL)
    @Positive
        public TypeKind getKind();

    @Positive
        @Override
    @Positive
        @DefinedBy(Api.LANGUAGE_MODEL)
    @Positive
        public <R, P> R accept(TypeVisitor<R, P> v, P p);
    @Positive
    }

    @Positive
    public static class ArrayType extends Type implements LoadableConstant, javax.lang.model.type.ArrayType {

    @Positive
        public Type elemtype;

    @Positive
        public ArrayType(Type elemtype, TypeSymbol arrayClass) {
    @Positive
        }

    @Positive
        public ArrayType(Type elemtype, TypeSymbol arrayClass, TypeMetadata metadata) {
    @Positive
        }

    @Positive
        public ArrayType(ArrayType that) {
    @Positive
        }

    @Positive
        public int poolTag();

    @Positive
        @Override
    @Positive
        public ArrayType cloneWithMetadata(TypeMetadata md);

    @Positive
        @Override
    @Positive
        public TypeTag getTag();

    @Positive
        public <R, S> R accept(Type.Visitor<R, S> v, S s);

    @Positive
        @DefinedBy(Api.LANGUAGE_MODEL)
    @Positive
        public String toString();

    @Positive
        @Override
    @Positive
        @DefinedBy(Api.LANGUAGE_MODEL)
    @Positive
        public boolean equals(Object obj);

    @Positive
        @DefinedBy(Api.LANGUAGE_MODEL)
    @Positive
        public int hashCode();

    @Positive
        public boolean isVarargs();

    @Positive
        public List<Type> allparams();

    @Positive
        public boolean isErroneous();

    @Positive
        public boolean isParameterized();

    @Positive
        @Override
    @Positive
        public boolean isReference();

    @Positive
        @Override
    @Positive
        public boolean isNullOrReference();

    @Positive
        public boolean isRaw();

    @Positive
        public ArrayType makeVarargs();

    @Positive
        @Pure
    @Positive
        public boolean contains(Type elem);

    @Positive
        public void complete();

    @Positive
        @DefinedBy(Api.LANGUAGE_MODEL)
    @Positive
        public Type getComponentType();

    @Positive
        @DefinedBy(Api.LANGUAGE_MODEL)
    @Positive
        public TypeKind getKind();

    @Positive
        @DefinedBy(Api.LANGUAGE_MODEL)
    @Positive
        public <R, P> R accept(TypeVisitor<R, P> v, P p);
    @Positive
    }

    @Positive
    public static class MethodType extends Type implements ExecutableType, LoadableConstant {

    @Positive
        public List<Type> argtypes;

    @Positive
        public Type restype;

    @Positive
        public List<Type> thrown;

    @Positive
        public Type recvtype;

    @Positive
        public MethodType(List<Type> argtypes, Type restype, List<Type> thrown, TypeSymbol methodClass) {
    @Positive
        }

    @Positive
        @Override
    @Positive
        public MethodType cloneWithMetadata(TypeMetadata md);

    @Positive
        @Override
    @Positive
        public TypeTag getTag();

    @Positive
        public <R, S> R accept(Type.Visitor<R, S> v, S s);

    @Positive
        @DefinedBy(Api.LANGUAGE_MODEL)
    @Positive
        public String toString();

    @Positive
        @DefinedBy(Api.LANGUAGE_MODEL)
    @Positive
        public List<Type> getParameterTypes();

    @Positive
        @DefinedBy(Api.LANGUAGE_MODEL)
    @Positive
        public Type getReturnType();

    @Positive
        @DefinedBy(Api.LANGUAGE_MODEL)
    @Positive
        public Type getReceiverType();

    @Positive
        @DefinedBy(Api.LANGUAGE_MODEL)
    @Positive
        public List<Type> getThrownTypes();

    @Positive
        public boolean isErroneous();

    @Positive
        @Override
    @Positive
        public int poolTag();

    @Positive
        @Pure
    @Positive
        public boolean contains(Type elem);

    @Positive
        public MethodType asMethodType();

    @Positive
        public void complete();

    @Positive
        @DefinedBy(Api.LANGUAGE_MODEL)
    @Positive
        public List<TypeVar> getTypeVariables();

    @Positive
        public TypeSymbol asElement();

    @Positive
        @DefinedBy(Api.LANGUAGE_MODEL)
    @Positive
        public TypeKind getKind();

    @Positive
        @DefinedBy(Api.LANGUAGE_MODEL)
    @Positive
        public <R, P> R accept(TypeVisitor<R, P> v, P p);
    @Positive
    }

    @Positive
    public static class PackageType extends Type implements NoType {

    @Positive
        @Override
    @Positive
        public PackageType cloneWithMetadata(TypeMetadata md);

    @Positive
        @Override
    @Positive
        public TypeTag getTag();

    @Positive
        @Override
    @Positive
        public <R, S> R accept(Type.Visitor<R, S> v, S s);

    @Positive
        @DefinedBy(Api.LANGUAGE_MODEL)
    @Positive
        public String toString();

    @Positive
        @DefinedBy(Api.LANGUAGE_MODEL)
    @Positive
        public TypeKind getKind();

    @Positive
        @DefinedBy(Api.LANGUAGE_MODEL)
    @Positive
        public <R, P> R accept(TypeVisitor<R, P> v, P p);
    @Positive
    }

    @Positive
    public static class ModuleType extends Type implements NoType {

    @Positive
        @Override
    @Positive
        public ModuleType cloneWithMetadata(TypeMetadata md);

    @Positive
        @Override
    @Positive
        public ModuleType annotatedType(List<Attribute.TypeCompound> annos);

    @Positive
        @Override
    @Positive
        public TypeTag getTag();

    @Positive
        @Override
    @Positive
        public <R, S> R accept(Type.Visitor<R, S> v, S s);

    @Positive
        @Override
    @Positive
        @DefinedBy(Api.LANGUAGE_MODEL)
    @Positive
        public String toString();

    @Positive
        @Override
    @Positive
        @DefinedBy(Api.LANGUAGE_MODEL)
    @Positive
        public TypeKind getKind();

    @Positive
        @Override
    @Positive
        @DefinedBy(Api.LANGUAGE_MODEL)
    @Positive
        public <R, P> R accept(TypeVisitor<R, P> v, P p);
    @Positive
    }

    @Positive
    public static class TypeVar extends Type implements TypeVariable {

    @Positive
        public Type lower;

    @Positive
        public TypeVar(Name name, Symbol owner, Type lower) {
    @Positive
        }

    @Positive
        public TypeVar(TypeSymbol tsym, Type bound, Type lower) {
    @Positive
        }

    @Positive
        public TypeVar(TypeSymbol tsym, Type bound, Type lower, TypeMetadata metadata) {
    @Positive
        }

    @Positive
        @Override
    @Positive
        public TypeVar cloneWithMetadata(TypeMetadata md);

    @Positive
        @Override
    @Positive
        public TypeTag getTag();

    @Positive
        @Override
    @Positive
        public <R, S> R accept(Type.Visitor<R, S> v, S s);

    @Positive
        @Override
    @Positive
        @DefinedBy(Api.LANGUAGE_MODEL)
    @Positive
        public Type getUpperBound();

    @Positive
        public void setUpperBound(Type bound);

    @Positive
        @Override
    @Positive
        @DefinedBy(Api.LANGUAGE_MODEL)
    @Positive
        public Type getLowerBound();

    @Positive
        @DefinedBy(Api.LANGUAGE_MODEL)
    @Positive
        public TypeKind getKind();

    @Positive
        public boolean isCaptured();

    @Positive
        @Override
    @Positive
        public boolean isReference();

    @Positive
        @Override
    @Positive
        public boolean isNullOrReference();

    @Positive
        @Override
    @Positive
        @DefinedBy(Api.LANGUAGE_MODEL)
    @Positive
        public <R, P> R accept(TypeVisitor<R, P> v, P p);
    @Positive
    }

    @Positive
    public static class CapturedType extends TypeVar {

    @Positive
        public WildcardType wildcard;

    @Positive
        public CapturedType(Name name, Symbol owner, Type upper, Type lower, WildcardType wildcard) {
    @Positive
        }

    @Positive
        public CapturedType(TypeSymbol tsym, Type bound, Type upper, Type lower, WildcardType wildcard, TypeMetadata metadata) {
    @Positive
        }

    @Positive
        @Override
    @Positive
        public CapturedType cloneWithMetadata(TypeMetadata md);

    @Positive
        @Override
    @Positive
        public <R, S> R accept(Type.Visitor<R, S> v, S s);

    @Positive
        @Override
    @Positive
        public boolean isCaptured();

    @Positive
        @Override
    @Positive
        @DefinedBy(Api.LANGUAGE_MODEL)
    @Positive
        public String toString();
    @Positive
    }

    @Positive
    public static abstract class DelegatedType extends Type {

    @Positive
        public Type qtype;

    @Positive
        public TypeTag tag;

    @Positive
        public DelegatedType(TypeTag tag, Type qtype) {
    @Positive
        }

    @Positive
        public DelegatedType(TypeTag tag, Type qtype, TypeMetadata metadata) {
    @Positive
        }

    @Positive
        public TypeTag getTag();

    @Positive
        @DefinedBy(Api.LANGUAGE_MODEL)
    @Positive
        public String toString();

    @Positive
        public List<Type> getTypeArguments();

    @Positive
        public Type getEnclosingType();

    @Positive
        public List<Type> getParameterTypes();

    @Positive
        public Type getReturnType();

    @Positive
        public Type getReceiverType();

    @Positive
        public List<Type> getThrownTypes();

    @Positive
        public List<Type> allparams();

    @Positive
        public Type getUpperBound();

    @Positive
        public boolean isErroneous();
    @Positive
    }

    @Positive
    public static class ForAll extends DelegatedType implements ExecutableType {

    @Positive
        public List<Type> tvars;

    @Positive
        public ForAll(List<Type> tvars, Type qtype) {
    @Positive
        }

    @Positive
        @Override
    @Positive
        public ForAll cloneWithMetadata(TypeMetadata md);

    @Positive
        @Override
    @Positive
        public <R, S> R accept(Type.Visitor<R, S> v, S s);

    @Positive
        @DefinedBy(Api.LANGUAGE_MODEL)
    @Positive
        public String toString();

    @Positive
        public List<Type> getTypeArguments();

    @Positive
        public boolean isErroneous();

    @Positive
        @Pure
    @Positive
        public boolean contains(Type elem);

    @Positive
        public MethodType asMethodType();

    @Positive
        public void complete();

    @Positive
        @DefinedBy(Api.LANGUAGE_MODEL)
    @Positive
        public List<TypeVar> getTypeVariables();

    @Positive
        @DefinedBy(Api.LANGUAGE_MODEL)
    @Positive
        public TypeKind getKind();

    @Positive
        @DefinedBy(Api.LANGUAGE_MODEL)
    @Positive
        public <R, P> R accept(TypeVisitor<R, P> v, P p);
    @Positive
    }

    @Positive
    public static class UndetVar extends DelegatedType {

    @Positive
        public interface UndetVarListener {

    @Positive
            void varBoundChanged(UndetVar uv, InferenceBound ib, Type bound, boolean update);

    @Positive
            default void varInstantiated(UndetVar uv);
    @Positive
        }

    @Positive
        public enum InferenceBound {

    @Positive
            LOWER {

    @Positive
                public InferenceBound complement();
    @Positive
            }
    @Positive
            , EQ {

    @Positive
                public InferenceBound complement();
    @Positive
            }
    @Positive
            , UPPER {

    @Positive
                public InferenceBound complement();
    @Positive
            }
    @Positive
            ;

    @Positive
            public abstract InferenceBound complement();

    @Positive
            public boolean lessThan(InferenceBound that);
    @Positive
        }

    @Positive
        public ArrayDeque<IncorporationAction> incorporationActions;

    @Positive
        protected Map<InferenceBound, List<Type>> bounds;

    @Positive
        public int declaredCount;

    @Positive
        public UndetVarListener listener;

    @Positive
        @Override
    @Positive
        public <R, S> R accept(Type.Visitor<R, S> v, S s);

    @Positive
        public UndetVar(TypeVar origin, UndetVarListener listener, Types types) {
    @Positive
        }

    @Positive
        @DefinedBy(Api.LANGUAGE_MODEL)
    @Positive
        public String toString();

    @Positive
        public String debugString();

    @Positive
        public void setThrow();

    @Positive
        public UndetVar dup(Types types);

    @Positive
        public void dupTo(UndetVar uv2, Types types);

    @Positive
        @Override
    @Positive
        public UndetVar cloneWithMetadata(TypeMetadata md);

    @Positive
        @Override
    @Positive
        public boolean isPartial();

    @Positive
        @Override
    @Positive
        public Type baseType();

    @Positive
        public Type getInst();

    @Positive
        public void setInst(Type inst);

    @Positive
        public List<Type> getBounds(InferenceBound... ibs);

    @Positive
        public List<Type> getDeclaredBounds();

    @Positive
        public void setBounds(InferenceBound ib, List<Type> newBounds);

    @Positive
        public final void addBound(InferenceBound ib, Type bound, Types types);

    @Positive
        public void substBounds(List<Type> from, List<Type> to, Types types);

    @Positive
        public final boolean isCaptured();

    @Positive
        public final boolean isThrows();
    @Positive
    }

    @Positive
    public static class JCNoType extends Type implements NoType {

    @Positive
        public JCNoType() {
    @Positive
        }

    @Positive
        @Override
    @Positive
        public JCNoType cloneWithMetadata(TypeMetadata md);

    @Positive
        @Override
    @Positive
        public TypeTag getTag();

    @Positive
        @Override
    @Positive
        @DefinedBy(Api.LANGUAGE_MODEL)
    @Positive
        public TypeKind getKind();

    @Positive
        @Override
    @Positive
        @DefinedBy(Api.LANGUAGE_MODEL)
    @Positive
        public <R, P> R accept(TypeVisitor<R, P> v, P p);

    @Positive
        @Override
    @Positive
        public boolean isCompound();
    @Positive
    }

    @Positive
    public static class JCVoidType extends Type implements NoType {

    @Positive
        public JCVoidType() {
    @Positive
        }

    @Positive
        @Override
    @Positive
        public JCVoidType cloneWithMetadata(TypeMetadata md);

    @Positive
        @Override
    @Positive
        public TypeTag getTag();

    @Positive
        @Override
    @Positive
        @DefinedBy(Api.LANGUAGE_MODEL)
    @Positive
        public TypeKind getKind();

    @Positive
        @Override
    @Positive
        public boolean isCompound();

    @Positive
        @Override
    @Positive
        @DefinedBy(Api.LANGUAGE_MODEL)
    @Positive
        public <R, P> R accept(TypeVisitor<R, P> v, P p);

    @Positive
        @Override
    @Positive
        public boolean isPrimitiveOrVoid();
    @Positive
    }

    @Positive
    static class BottomType extends Type implements NullType {

    @Positive
        public BottomType() {
    @Positive
        }

    @Positive
        @Override
    @Positive
        public BottomType cloneWithMetadata(TypeMetadata md);

    @Positive
        @Override
    @Positive
        public TypeTag getTag();

    @Positive
        @Override
    @Positive
        @DefinedBy(Api.LANGUAGE_MODEL)
    @Positive
        public TypeKind getKind();

    @Positive
        @Override
    @Positive
        public boolean isCompound();

    @Positive
        @Override
    @Positive
        @DefinedBy(Api.LANGUAGE_MODEL)
    @Positive
        public <R, P> R accept(TypeVisitor<R, P> v, P p);

    @Positive
        @Override
    @Positive
        public Type constType(Object value);

    @Positive
        @Override
    @Positive
        public String stringValue();

    @Positive
        @Override
    @Positive
        public boolean isNullOrReference();
    @Positive
    }

    @Positive
    public static class ErrorType extends ClassType implements javax.lang.model.type.ErrorType {

    @Positive
        public ErrorType(ClassSymbol c, Type originalType) {
    @Positive
        }

    @Positive
        public ErrorType(Type originalType, TypeSymbol tsym) {
    @Positive
        }

    @Positive
        @Override
    @Positive
        public ErrorType cloneWithMetadata(TypeMetadata md);

    @Positive
        @Override
    @Positive
        public TypeTag getTag();

    @Positive
        @Override
    @Positive
        public boolean isPartial();

    @Positive
        @Override
    @Positive
        public boolean isReference();

    @Positive
        @Override
    @Positive
        public boolean isNullOrReference();

    @Positive
        public ErrorType(Name name, TypeSymbol container, Type originalType) {
    @Positive
        }

    @Positive
        @Override
    @Positive
        public <R, S> R accept(Type.Visitor<R, S> v, S s);

    @Positive
        public Type constType(Object constValue);

    @Positive
        @DefinedBy(Api.LANGUAGE_MODEL)
    @Positive
        public Type getEnclosingType();

    @Positive
        public Type getReturnType();

    @Positive
        public Type asSub(Symbol sym);

    @Positive
        public boolean isGenType(Type t);

    @Positive
        public boolean isErroneous();

    @Positive
        public boolean isCompound();

    @Positive
        public boolean isInterface();

    @Positive
        public List<Type> allparams();

    @Positive
        @DefinedBy(Api.LANGUAGE_MODEL)
    @Positive
        public List<Type> getTypeArguments();

    @Positive
        @DefinedBy(Api.LANGUAGE_MODEL)
    @Positive
        public TypeKind getKind();

    @Positive
        public Type getOriginalType();

    @Positive
        @DefinedBy(Api.LANGUAGE_MODEL)
    @Positive
        public <R, P> R accept(TypeVisitor<R, P> v, P p);
    @Positive
    }

    @Positive
    public static class UnknownType extends Type {

    @Positive
        public UnknownType() {
    @Positive
        }

    @Positive
        @Override
    @Positive
        public UnknownType cloneWithMetadata(TypeMetadata md);

    @Positive
        @Override
    @Positive
        public TypeTag getTag();

    @Positive
        @Override
    @Positive
        @DefinedBy(Api.LANGUAGE_MODEL)
    @Positive
        public <R, P> R accept(TypeVisitor<R, P> v, P p);

    @Positive
        @Override
    @Positive
        public boolean isPartial();
    @Positive
    }

    @Positive
    public interface Visitor<R, S> {

    @Positive
        R visitClassType(ClassType t, S s);

    @Positive
        R visitWildcardType(WildcardType t, S s);

    @Positive
        R visitArrayType(ArrayType t, S s);

    @Positive
        R visitMethodType(MethodType t, S s);

    @Positive
        R visitPackageType(PackageType t, S s);

    @Positive
        R visitModuleType(ModuleType t, S s);

    @Positive
        R visitTypeVar(TypeVar t, S s);

    @Positive
        R visitCapturedType(CapturedType t, S s);

    @Positive
        R visitForAll(ForAll t, S s);

    @Positive
        R visitUndetVar(UndetVar t, S s);

    @Positive
        R visitErrorType(ErrorType t, S s);

    @Positive
        R visitType(Type t, S s);
    @Positive
    }
    @Positive
}
