/*
 * CFWR semantic augmentation: applied semantic-preserving transformations.
 */
/*
    @Positive
 * Copyright (c) 2006, 2017, Oracle and/or its affiliates. All rights reserved.
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
import org.checkerframework.dataflow.qual.Pure;
    @Positive
import java.io.IOException;
    @Positive
import java.net.URI;
    @Positive
import java.util.ArrayList;
    @Positive
import java.util.Collections;
    @Positive
import java.util.List;
    @Positive
import java.util.Set;
    @Positive
import javax.tools.*;
    @Positive
import javax.tools.JavaFileObject.Kind;
    @Positive
import com.sun.tools.javac.util.ClientCodeException;
    @Positive
import com.sun.tools.javac.util.DefinedBy;
    @Positive
import com.sun.tools.javac.util.DefinedBy.Api;

    @Positive
public class WrappingJavaFileManager<M extends JavaFileManager> extends ForwardingJavaFileManager<M> {

    @Positive
    protected WrappingJavaFileManager(M fileManager) {
    @Positive
    }

    @Positive
    protected FileObject wrap(FileObject fileObject);

    @Positive
    protected JavaFileObject wrap(JavaFileObject fileObject);

    @Positive
    protected FileObject unwrap(FileObject fileObject);

    @Positive
    protected JavaFileObject unwrap(JavaFileObject fileObject);

    @Positive
    protected Iterable<JavaFileObject> wrap(Iterable<JavaFileObject> fileObjects);

    @Positive
    protected URI unwrap(URI uri);

    @Positive
    @DefinedBy(Api.COMPILER)
    @Positive
    public Iterable<JavaFileObject> list(Location location, String packageName, Set<Kind> kinds, boolean recurse) throws IOException;

    @Positive
    @DefinedBy(Api.COMPILER)
    @Positive
    public String inferBinaryName(Location location, JavaFileObject file);

    @Positive
    @DefinedBy(Api.COMPILER)
    @Positive
    public JavaFileObject getJavaFileForInput(Location location, String className, Kind kind) throws IOException;

    @Positive
    @DefinedBy(Api.COMPILER)
    @Positive
    public JavaFileObject getJavaFileForOutput(Location location, String className, Kind kind, FileObject sibling) throws IOException;

    @Positive
    @DefinedBy(Api.COMPILER)
    @Positive
    public FileObject getFileForInput(Location location, String packageName, String relativeName) throws IOException;

    @Positive
    @DefinedBy(Api.COMPILER)
    @Positive
    public FileObject getFileForOutput(Location location, String packageName, String relativeName, FileObject sibling) throws IOException;

    @Positive
    @Override
    @Positive
    @DefinedBy(Api.COMPILER)
    @Positive
    @Pure
    @Positive
    public boolean contains(Location location, FileObject file) throws IOException;
    @Positive
}
