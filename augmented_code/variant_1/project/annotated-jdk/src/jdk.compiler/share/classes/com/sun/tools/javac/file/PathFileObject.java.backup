/*
 * Copyright (c) 2009, 2021, Oracle and/or its affiliates. All rights reserved.
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
package com.sun.tools.javac.file;

import org.checkerframework.checker.nullness.qual.EnsuresNonNullIf;
import org.checkerframework.checker.nullness.qual.NonNull;
import org.checkerframework.checker.nullness.qual.Nullable;
import org.checkerframework.dataflow.qual.Pure;
import org.checkerframework.dataflow.qual.SideEffectFree;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.io.OutputStream;
import java.io.OutputStreamWriter;
import java.io.Reader;
import java.io.Writer;
import java.net.URI;
import java.net.URISyntaxException;
import java.nio.ByteBuffer;
import java.nio.CharBuffer;
import java.nio.charset.CharsetDecoder;
import java.nio.file.FileSystem;
import java.nio.file.FileSystems;
import java.nio.file.Files;
import java.nio.file.LinkOption;
import java.nio.file.Path;
import java.text.Normalizer;
import java.util.Objects;
import javax.lang.model.element.Modifier;
import javax.lang.model.element.NestingKind;
import javax.tools.FileObject;
import javax.tools.JavaFileObject;
import com.sun.tools.javac.file.RelativePath.RelativeFile;
import com.sun.tools.javac.util.DefinedBy;
import com.sun.tools.javac.util.DefinedBy.Api;

public abstract class PathFileObject implements JavaFileObject {

    protected final BaseFileManager fileManager;

    protected final Path path;

    static PathFileObject forDirectoryPath(BaseFileManager fileManager, Path path, Path userPackageRootDir, RelativePath relativePath);

    private static class DirectoryFileObject extends PathFileObject {

        @Override
        @DefinedBy(Api.COMPILER)
        public String getName();

        @Override
        public String inferBinaryName(Iterable<? extends Path> paths);

        @Override
        public String toString();

        @Override
        PathFileObject getSibling(String baseName);
    }

    public static PathFileObject forJarPath(BaseFileManager fileManager, Path path, Path userJarPath);

    private static class JarFileObject extends PathFileObject {

        @Override
        @DefinedBy(Api.COMPILER)
        public String getName();

        @Override
        public String inferBinaryName(Iterable<? extends Path> paths);

        @Override
        @DefinedBy(Api.COMPILER)
        public URI toUri();

        @Override
        public String toString();

        @Override
        PathFileObject getSibling(String baseName);
    }

    public static PathFileObject forJRTPath(BaseFileManager fileManager, final Path path);

    private static class JRTFileObject extends PathFileObject {

        @Override
        @DefinedBy(Api.COMPILER)
        public String getName();

        @Override
        public String inferBinaryName(Iterable<? extends Path> paths);

        @Override
        public String toString();

        @Override
        PathFileObject getSibling(String baseName);
    }

    static PathFileObject forSimplePath(BaseFileManager fileManager, Path path, Path userPath);

    private static class SimpleFileObject extends PathFileObject {

        @Override
        @DefinedBy(Api.COMPILER)
        public String getName();

        @Override
        @DefinedBy(Api.COMPILER)
        public String getShortName();

        @Override
        public String inferBinaryName(Iterable<? extends Path> paths);

        @Override
        @DefinedBy(Api.COMPILER)
        public Kind getKind();

        @Override
        @DefinedBy(Api.COMPILER)
        public boolean isNameCompatible(String simpleName, Kind kind);

        @Override
        @DefinedBy(Api.COMPILER)
        public URI toUri();

        @Override
        PathFileObject getSibling(String baseName);
    }

    protected PathFileObject(BaseFileManager fileManager, Path path) {
    }

    abstract String inferBinaryName(Iterable<? extends Path> paths);

    abstract PathFileObject getSibling(String basename);

    public Path getPath();

    public String getShortName();

    @Override
    @DefinedBy(Api.COMPILER)
    public Kind getKind();

    @Override
    @DefinedBy(Api.COMPILER)
    public boolean isNameCompatible(String simpleName, Kind kind);

    protected boolean isPathNameCompatible(Path p, String simpleName, Kind kind);

    @Override
    @DefinedBy(Api.COMPILER)
    public NestingKind getNestingKind();

    @Override
    @DefinedBy(Api.COMPILER)
    public Modifier getAccessLevel();

    @Override
    @DefinedBy(Api.COMPILER)
    public URI toUri();

    @Override
    @DefinedBy(Api.COMPILER)
    public InputStream openInputStream() throws IOException;

    @Override
    @DefinedBy(Api.COMPILER)
    public OutputStream openOutputStream() throws IOException;

    @Override
    @DefinedBy(Api.COMPILER)
    public Reader openReader(boolean ignoreEncodingErrors) throws IOException;

    @Override
    @DefinedBy(Api.COMPILER)
    public CharSequence getCharContent(boolean ignoreEncodingErrors) throws IOException;

    @Override
    @DefinedBy(Api.COMPILER)
    public Writer openWriter() throws IOException;

    @Override
    @DefinedBy(Api.COMPILER)
    public long getLastModified();

    @Override
    @DefinedBy(Api.COMPILER)
    public boolean delete();

    boolean isSameFile(PathFileObject other);

    @Override
    @Pure
    @EnsuresNonNullIf(expression = "#1", result = true)
    public boolean equals(@Nullable Object other);

    @Override
    public int hashCode();

    @Override
    public String toString();

    protected static String toBinaryName(RelativePath relativePath);

    protected static String toBinaryName(Path relativePath);

    public static String getSimpleName(FileObject fo);

    public static class CannotCreateUriError extends Error {

        public CannotCreateUriError(String value, Throwable cause) {
        }
    }
}
