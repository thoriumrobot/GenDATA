/*
    @Positive
 * Copyright (c) 2006, 2021, Oracle and/or its affiliates. All rights reserved.
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
package com.sun.tools.javac.api;

    @Positive
import org.checkerframework.checker.nullness.qual.EnsuresNonNullIf;
    @Positive
import org.checkerframework.checker.nullness.qual.NonNull;
    @Positive
import org.checkerframework.checker.nullness.qual.Nullable;
    @Positive
import org.checkerframework.dataflow.qual.Pure;
    @Positive
import org.checkerframework.dataflow.qual.SideEffectFree;
    @Positive
import java.util.function.Predicate;
    @Positive
import javax.lang.model.element.Element;
    @Positive
import javax.lang.model.element.ExecutableElement;
    @Positive
import javax.lang.model.element.TypeElement;
    @Positive
import com.sun.tools.javac.code.Kinds.Kind;
    @Positive
import com.sun.tools.javac.code.Scope.CompoundScope;
    @Positive
import com.sun.tools.javac.code.Symbol;
    @Positive
import com.sun.tools.javac.comp.AttrContext;
    @Positive
import com.sun.tools.javac.comp.Env;
    @Positive
import com.sun.tools.javac.util.DefinedBy;
    @Positive
import com.sun.tools.javac.util.DefinedBy.Api;
    @Positive
import com.sun.tools.javac.util.Assert;

    @Positive
public class JavacScope implements com.sun.source.tree.Scope {

    @Positive
    static JavacScope create(Env<AttrContext> env);

    @Positive
    protected final Env<AttrContext> env;

    @Positive
    @DefinedBy(Api.COMPILER_TREE)
    @Positive
    public JavacScope getEnclosingScope();

    @Positive
    @DefinedBy(Api.COMPILER_TREE)
    @Positive
    public TypeElement getEnclosingClass();

    @Positive
    @DefinedBy(Api.COMPILER_TREE)
    @Positive
    public ExecutableElement getEnclosingMethod();

    @Positive
    @DefinedBy(Api.COMPILER_TREE)
    @Positive
    public Iterable<? extends Element> getLocalElements();

    @Positive
    public Env<AttrContext> getEnv();

    @Positive
    public boolean isStarImportScope();

    @Positive
    @Pure
    @Positive
    @EnsuresNonNullIf(expression = "#1", result = true)
    @Positive
    public boolean equals(@Nullable Object other);

    @Positive
    public int hashCode();

    @Positive
    public String toString();
    @Positive
}

// CFWR semantic augmentation - variant 1
