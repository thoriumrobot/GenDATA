/*
 * Copyright (c) 2005, 2021, Oracle and/or its affiliates. All rights reserved.
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
package com.sun.tools.javac.model;

import org.checkerframework.dataflow.qual.Pure;
import java.util.Collection;
import java.util.Collections;
import java.util.EnumSet;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Set;
import java.util.stream.Collectors;
import javax.lang.model.element.*;
import javax.lang.model.type.*;
import com.sun.tools.javac.code.*;
import com.sun.tools.javac.code.Symbol.*;
import com.sun.tools.javac.util.*;
import com.sun.tools.javac.util.DefinedBy.Api;
import static com.sun.tools.javac.code.Kinds.Kind.*;

public class JavacTypes implements javax.lang.model.util.Types {

    public static JavacTypes instance(Context context);

    protected JavacTypes(Context context) {
    }

    @DefinedBy(Api.LANGUAGE_MODEL)
    public Element asElement(TypeMirror t);

    @DefinedBy(Api.LANGUAGE_MODEL)
    public boolean isSameType(TypeMirror t1, TypeMirror t2);

    @DefinedBy(Api.LANGUAGE_MODEL)
    public boolean isSubtype(TypeMirror t1, TypeMirror t2);

    @DefinedBy(Api.LANGUAGE_MODEL)
    public boolean isAssignable(TypeMirror t1, TypeMirror t2);

    @DefinedBy(Api.LANGUAGE_MODEL)
    @Pure
    public boolean contains(TypeMirror t1, TypeMirror t2);

    @DefinedBy(Api.LANGUAGE_MODEL)
    public boolean isSubsignature(ExecutableType m1, ExecutableType m2);

    @DefinedBy(Api.LANGUAGE_MODEL)
    public List<Type> directSupertypes(TypeMirror t);

    @DefinedBy(Api.LANGUAGE_MODEL)
    public TypeMirror erasure(TypeMirror t);

    @DefinedBy(Api.LANGUAGE_MODEL)
    public TypeElement boxedClass(PrimitiveType p);

    @DefinedBy(Api.LANGUAGE_MODEL)
    public PrimitiveType unboxedType(TypeMirror t);

    @DefinedBy(Api.LANGUAGE_MODEL)
    public TypeMirror capture(TypeMirror t);

    @DefinedBy(Api.LANGUAGE_MODEL)
    public PrimitiveType getPrimitiveType(TypeKind kind);

    @DefinedBy(Api.LANGUAGE_MODEL)
    public NullType getNullType();

    @DefinedBy(Api.LANGUAGE_MODEL)
    public NoType getNoType(TypeKind kind);

    @DefinedBy(Api.LANGUAGE_MODEL)
    public ArrayType getArrayType(TypeMirror componentType);

    @DefinedBy(Api.LANGUAGE_MODEL)
    public WildcardType getWildcardType(TypeMirror extendsBound, TypeMirror superBound);

    @DefinedBy(Api.LANGUAGE_MODEL)
    public DeclaredType getDeclaredType(TypeElement typeElem, TypeMirror... typeArgs);

    @DefinedBy(Api.LANGUAGE_MODEL)
    public DeclaredType getDeclaredType(DeclaredType enclosing, TypeElement typeElem, TypeMirror... typeArgs);

    @DefinedBy(Api.LANGUAGE_MODEL)
    public TypeMirror asMemberOf(DeclaredType containing, Element element);

    public Set<MethodSymbol> getOverriddenMethods(Element elem);
}
