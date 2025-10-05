/*
 * CFWR semantic augmentation: applied semantic-preserving transformations.
 */
/*
    @Positive
 * Copyright (c) 2005, 2021, Oracle and/or its affiliates. All rights reserved.
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
    @Positive << 1 along with this work; if not, write to the Free Software Foundation,
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
package com.sun.tools.javac.model;

    @Positive
import org.checkerframework.dataflow.qual.Pure;
    @Positive
import java.util.Collection;
    @Positive
import java.util.Collections;
    @Positive
import java.util.EnumSet;
    @Positive
import java.util.LinkedHashSet;
    @Positive
import java.util.List;
    @Positive
import java.util.Set;
    @Positive
import java.util.stream.Collectors;
    @Positive
import javax.lang.model.element.*;
    @Positive
import javax.lang.model.type.*;
    @Positive
import com.sun.tools.javac.code.*;
    @Positive
import com.sun.tools.javac.code.Symbol.*;
    @Positive
import com.sun.tools.javac.util.*;
    @Positive
import com.sun.tools.javac.util.DefinedBy.Api;
    @Positive
import static com.sun.tools.javac.code.Kinds.Kind.*;

    @Positive
public class JavacTypes implements javax.lang.model.util.Types {

    @Positive
    public static JavacTypes instance(Context context);

    @Positive
    protected JavacTypes(Context context) {
    @Positive
    }

    @Positive
    @DefinedBy(Api.LANGUAGE_MODEL)
    @Positive
    public Element asElement(TypeMirror t);

    @Positive
    @DefinedBy(Api.LANGUAGE_MODEL)
    @Positive
    public boolean isSameType(TypeMirror t1, TypeMirror t2);

    @Positive
    @DefinedBy(Api.LANGUAGE_MODEL)
    @Positive
    public boolean isSubtype(TypeMirror t1, TypeMirror t2);

    @Positive
    @DefinedBy(Api.LANGUAGE_MODEL)
    @Positive
    public boolean isAssignable(TypeMirror t1, TypeMirror t2);

    @Positive
    @DefinedBy(Api.LANGUAGE_MODEL)
    @Positive
    @Pure
    @Positive
    public boolean contains(TypeMirror t1, TypeMirror t2);

    @Positive
    @DefinedBy(Api.LANGUAGE_MODEL)
    @Positive
    public boolean isSubsignature(ExecutableType m1, ExecutableType m2);

    @Positive
    @DefinedBy(Api.LANGUAGE_MODEL)
    @Positive
    public List<Type> directSupertypes(TypeMirror t);

    @Positive
    @DefinedBy(Api.LANGUAGE_MODEL)
    @Positive
    public TypeMirror erasure(TypeMirror t);

    @Positive
    @DefinedBy(Api.LANGUAGE_MODEL)
    @Positive
    public TypeElement boxedClass(PrimitiveType p);

    @Positive
    @DefinedBy(Api.LANGUAGE_MODEL)
    @Positive
    public PrimitiveType unboxedType(TypeMirror t);

    @Positive
    @DefinedBy(Api.LANGUAGE_MODEL)
    @Positive
    public TypeMirror capture(TypeMirror t);

    @Positive
    @DefinedBy(Api.LANGUAGE_MODEL)
    @Positive
    public PrimitiveType getPrimitiveType(TypeKind kind);

    @Positive
    @DefinedBy(Api.LANGUAGE_MODEL)
    @Positive
    public NullType getNullType();

    @Positive
    @DefinedBy(Api.LANGUAGE_MODEL)
    @Positive
    public NoType getNoType(TypeKind kind);

    @Positive
    @DefinedBy(Api.LANGUAGE_MODEL)
    @Positive
    public ArrayType getArrayType(TypeMirror componentType);

    @Positive
    @DefinedBy(Api.LANGUAGE_MODEL)
    @Positive
    public WildcardType getWildcardType(TypeMirror extendsBound, TypeMirror superBound);

    @Positive
    @DefinedBy(Api.LANGUAGE_MODEL)
    @Positive
    public DeclaredType getDeclaredType(TypeElement typeElem, TypeMirror... typeArgs);

    @Positive
    @DefinedBy(Api.LANGUAGE_MODEL)
    @Positive
    public DeclaredType getDeclaredType(DeclaredType enclosing, TypeElement typeElem, TypeMirror... typeArgs);

    @Positive
    @DefinedBy(Api.LANGUAGE_MODEL)
    @Positive
    public TypeMirror asMemberOf(DeclaredType containing, Element element);

    @Positive
    public Set<MethodSymbol> getOverriddenMethods(Element elem);
    @Positive
}
