/*
 * Copyright (c) 2005, 2020, Oracle and/or its affiliates. All rights reserved.
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
package javax.lang.model.util;

import org.checkerframework.checker.nullness.qual.Nullable;
import org.checkerframework.dataflow.qual.Pure;
import org.checkerframework.framework.qual.AnnotatedFor;
import java.lang.annotation.Annotation;
import java.lang.annotation.AnnotationTypeMismatchException;
import java.lang.annotation.IncompleteAnnotationException;
import java.util.List;
import javax.lang.model.element.*;
import javax.lang.model.type.*;

@AnnotatedFor("nullness")
public interface Types {

    @Nullable
    Element asElement(TypeMirror t);

    boolean isSameType(TypeMirror t1, TypeMirror t2);

    boolean isSubtype(TypeMirror t1, TypeMirror t2);

    boolean isAssignable(TypeMirror t1, TypeMirror t2);

    @Pure
    boolean contains(TypeMirror t1, TypeMirror t2);

    boolean isSubsignature(ExecutableType m1, ExecutableType m2);

    List<? extends TypeMirror> directSupertypes(TypeMirror t);

    TypeMirror erasure(TypeMirror t);

    TypeElement boxedClass(PrimitiveType p);

    PrimitiveType unboxedType(TypeMirror t);

    TypeMirror capture(TypeMirror t);

    PrimitiveType getPrimitiveType(TypeKind kind);

    NullType getNullType();

    NoType getNoType(TypeKind kind);

    ArrayType getArrayType(TypeMirror componentType);

    WildcardType getWildcardType(@Nullable TypeMirror extendsBound, @Nullable TypeMirror superBound);

    DeclaredType getDeclaredType(TypeElement typeElem, TypeMirror... typeArgs);

    DeclaredType getDeclaredType(@Nullable DeclaredType containing, TypeElement typeElem, TypeMirror... typeArgs);

    TypeMirror asMemberOf(DeclaredType containing, Element element);
}
