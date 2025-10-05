/*
 * Copyright (c) 2012, 2016, Oracle and/or its affiliates. All rights reserved.
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
package com.sun.tools.sjavac.comp;

import org.checkerframework.checker.nullness.qual.EnsuresNonNullIf;
import org.checkerframework.checker.nullness.qual.NonNull;
import org.checkerframework.checker.nullness.qual.Nullable;
import org.checkerframework.dataflow.qual.Pure;
import org.checkerframework.dataflow.qual.SideEffectFree;
import java.io.*;
import java.net.URI;
import java.nio.file.NoSuchFileException;
import javax.lang.model.element.Modifier;
import javax.lang.model.element.NestingKind;
import javax.tools.JavaFileObject;
import com.sun.tools.javac.util.DefinedBy;
import com.sun.tools.javac.util.DefinedBy.Api;

public class SmartFileObject implements JavaFileObject {

    public SmartFileObject(JavaFileObject r) {
    }

    @Override
    @Pure
    @EnsuresNonNullIf(expression = "#1", result = true)
    public boolean equals(@Nullable Object other);

    @Override
    public int hashCode();

    @DefinedBy(Api.COMPILER)
    public Kind getKind();

    @DefinedBy(Api.COMPILER)
    public boolean isNameCompatible(String simpleName, Kind kind);

    @DefinedBy(Api.COMPILER)
    public URI toUri();

    @DefinedBy(Api.COMPILER)
    public String getName();

    @DefinedBy(Api.COMPILER)
    public InputStream openInputStream() throws IOException;

    @DefinedBy(Api.COMPILER)
    public OutputStream openOutputStream() throws IOException;

    @DefinedBy(Api.COMPILER)
    public CharSequence getCharContent(boolean ignoreEncodingErrors) throws IOException;

    @DefinedBy(Api.COMPILER)
    public Writer openWriter() throws IOException;

    @DefinedBy(Api.COMPILER)
    public long getLastModified();

    @DefinedBy(Api.COMPILER)
    public boolean delete();

    @DefinedBy(Api.COMPILER)
    public Modifier getAccessLevel();

    @DefinedBy(Api.COMPILER)
    public NestingKind getNestingKind();

    @DefinedBy(Api.COMPILER)
    public Reader openReader(boolean ignoreEncodingErrors) throws IOException;
}
