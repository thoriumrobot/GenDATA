/*
 * Copyright (c) 2006, 2017, Oracle and/or its affiliates. All rights reserved.
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
package com.sun.tools.javac.api;

import org.checkerframework.dataflow.qual.Pure;
import java.io.IOException;
import java.net.URI;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Set;
import javax.tools.*;
import javax.tools.JavaFileObject.Kind;
import com.sun.tools.javac.util.ClientCodeException;
import com.sun.tools.javac.util.DefinedBy;
import com.sun.tools.javac.util.DefinedBy.Api;

public class WrappingJavaFileManager<M extends JavaFileManager> extends ForwardingJavaFileManager<M> {

    protected WrappingJavaFileManager(M fileManager) {
    }

    protected FileObject wrap(FileObject fileObject);

    protected JavaFileObject wrap(JavaFileObject fileObject);

    protected FileObject unwrap(FileObject fileObject);

    protected JavaFileObject unwrap(JavaFileObject fileObject);

    protected Iterable<JavaFileObject> wrap(Iterable<JavaFileObject> fileObjects);

    protected URI unwrap(URI uri);

    @DefinedBy(Api.COMPILER)
    public Iterable<JavaFileObject> list(Location location, String packageName, Set<Kind> kinds, boolean recurse) throws IOException;

    @DefinedBy(Api.COMPILER)
    public String inferBinaryName(Location location, JavaFileObject file);

    @DefinedBy(Api.COMPILER)
    public JavaFileObject getJavaFileForInput(Location location, String className, Kind kind) throws IOException;

    @DefinedBy(Api.COMPILER)
    public JavaFileObject getJavaFileForOutput(Location location, String className, Kind kind, FileObject sibling) throws IOException;

    @DefinedBy(Api.COMPILER)
    public FileObject getFileForInput(Location location, String packageName, String relativeName) throws IOException;

    @DefinedBy(Api.COMPILER)
    public FileObject getFileForOutput(Location location, String packageName, String relativeName, FileObject sibling) throws IOException;

    @Override
    @DefinedBy(Api.COMPILER)
    @Pure
    public boolean contains(Location location, FileObject file) throws IOException;
}
