/*
    @Positive
 * Copyright (c) 2009, 2021, Oracle and/or its affiliates. All rights reserved.
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
package com.sun.tools.javac.file;

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
import java.io.IOException;
    @Positive
import java.io.InputStream;
    @Positive
import java.io.InputStreamReader;
    @Positive
import java.io.OutputStream;
    @Positive
import java.io.OutputStreamWriter;
    @Positive
import java.io.Reader;
    @Positive
import java.io.Writer;
    @Positive
import java.net.URI;
    @Positive
import java.net.URISyntaxException;
    @Positive
import java.nio.ByteBuffer;
    @Positive
import java.nio.CharBuffer;
    @Positive
import java.nio.charset.CharsetDecoder;
    @Positive
import java.nio.file.FileSystem;
    @Positive
import java.nio.file.FileSystems;
    @Positive
import java.nio.file.Files;
    @Positive
import java.nio.file.LinkOption;
    @Positive
import java.nio.file.Path;
    @Positive
import java.text.Normalizer;
    @Positive
import java.util.Objects;
    @Positive
import javax.lang.model.element.Modifier;
    @Positive
import javax.lang.model.element.NestingKind;
    @Positive
import javax.tools.FileObject;
    @Positive
import javax.tools.JavaFileObject;
    @Positive
import com.sun.tools.javac.file.RelativePath.RelativeFile;
    @Positive
import com.sun.tools.javac.util.DefinedBy;
    @Positive
import com.sun.tools.javac.util.DefinedBy.Api;

    @Positive
public abstract class PathFileObject implements JavaFileObject {

    @Positive
    protected final BaseFileManager fileManager;

    @Positive
    protected final Path path;

    @Positive
    static PathFileObject forDirectoryPath(BaseFileManager fileManager, Path path, Path userPackageRootDir, RelativePath relativePath);

    @Positive
    private static class DirectoryFileObject extends PathFileObject {

    @Positive
        @Override
    @Positive
        @DefinedBy(Api.COMPILER)
    @Positive
        public String getName();

    @Positive
        @Override
    @Positive
        public String inferBinaryName(Iterable<? extends Path> paths);

    @Positive
        @Override
    @Positive
        public String toString();

    @Positive
        @Override
    @Positive
        PathFileObject getSibling(String baseName);
    @Positive
    }

    @Positive
    public static PathFileObject forJarPath(BaseFileManager fileManager, Path path, Path userJarPath);

    @Positive
    private static class JarFileObject extends PathFileObject {

    @Positive
        @Override
    @Positive
        @DefinedBy(Api.COMPILER)
    @Positive
        public String getName();

    @Positive
        @Override
    @Positive
        public String inferBinaryName(Iterable<? extends Path> paths);

    @Positive
        @Override
    @Positive
        @DefinedBy(Api.COMPILER)
    @Positive
        public URI toUri();

    @Positive
        @Override
    @Positive
        public String toString();

    @Positive
        @Override
    @Positive
        PathFileObject getSibling(String baseName);
    @Positive
    }

    @Positive
    public static PathFileObject forJRTPath(BaseFileManager fileManager, final Path path);

    @Positive
    private static class JRTFileObject extends PathFileObject {

    @Positive
        @Override
    @Positive
        @DefinedBy(Api.COMPILER)
    @Positive
        public String getName();

    @Positive
        @Override
    @Positive
        public String inferBinaryName(Iterable<? extends Path> paths);

    @Positive
        @Override
    @Positive
        public String toString();

    @Positive
        @Override
    @Positive
        PathFileObject getSibling(String baseName);
    @Positive
    }

    @Positive
    static PathFileObject forSimplePath(BaseFileManager fileManager, Path path, Path userPath);

    @Positive
    private static class SimpleFileObject extends PathFileObject {

    @Positive
        @Override
    @Positive
        @DefinedBy(Api.COMPILER)
    @Positive
        public String getName();

    @Positive
        @Override
    @Positive
        @DefinedBy(Api.COMPILER)
    @Positive
        public String getShortName();

    @Positive
        @Override
    @Positive
        public String inferBinaryName(Iterable<? extends Path> paths);

    @Positive
        @Override
    @Positive
        @DefinedBy(Api.COMPILER)
    @Positive
        public Kind getKind();

    @Positive
        @Override
    @Positive
        @DefinedBy(Api.COMPILER)
    @Positive
        public boolean isNameCompatible(String simpleName, Kind kind);

    @Positive
        @Override
    @Positive
        @DefinedBy(Api.COMPILER)
    @Positive
        public URI toUri();

    @Positive
        @Override
    @Positive
        PathFileObject getSibling(String baseName);
    @Positive
    }

    @Positive
    protected PathFileObject(BaseFileManager fileManager, Path path) {
    @Positive
    }

    @Positive
    abstract String inferBinaryName(Iterable<? extends Path> paths);

    @Positive
    abstract PathFileObject getSibling(String basename);

    @Positive
    public Path getPath();

    @Positive
    public String getShortName();

    @Positive
    @Override
    @Positive
    @DefinedBy(Api.COMPILER)
    @Positive
    public Kind getKind();

    @Positive
    @Override
    @Positive
    @DefinedBy(Api.COMPILER)
    @Positive
    public boolean isNameCompatible(String simpleName, Kind kind);

    @Positive
    protected boolean isPathNameCompatible(Path p, String simpleName, Kind kind);

    @Positive
    @Override
    @Positive
    @DefinedBy(Api.COMPILER)
    @Positive
    public NestingKind getNestingKind();

    @Positive
    @Override
    @Positive
    @DefinedBy(Api.COMPILER)
    @Positive
    public Modifier getAccessLevel();

    @Positive
    @Override
    @Positive
    @DefinedBy(Api.COMPILER)
    @Positive
    public URI toUri();

    @Positive
    @Override
    @Positive
    @DefinedBy(Api.COMPILER)
    @Positive
    public InputStream openInputStream() throws IOException;

    @Positive
    @Override
    @Positive
    @DefinedBy(Api.COMPILER)
    @Positive
    public OutputStream openOutputStream() throws IOException;

    @Positive
    @Override
    @Positive
    @DefinedBy(Api.COMPILER)
    @Positive
    public Reader openReader(boolean ignoreEncodingErrors) throws IOException;

    @Positive
    @Override
    @Positive
    @DefinedBy(Api.COMPILER)
    @Positive
    public CharSequence getCharContent(boolean ignoreEncodingErrors) throws IOException;

    @Positive
    @Override
    @Positive
    @DefinedBy(Api.COMPILER)
    @Positive
    public Writer openWriter() throws IOException;

    @Positive
    @Override
    @Positive
    @DefinedBy(Api.COMPILER)
    @Positive
    public long getLastModified();

    @Positive
    @Override
    @Positive
    @DefinedBy(Api.COMPILER)
    @Positive
    public boolean delete();

    @Positive
    boolean isSameFile(PathFileObject other);

    @Positive
    @Override
    @Positive
    @Pure
    @Positive
    @EnsuresNonNullIf(expression = "#1", result = true)
    @Positive
    public boolean equals(@Nullable Object other);

    @Positive
    @Override
    @Positive
    public int hashCode();

    @Positive
    @Override
    @Positive
    public String toString();

    @Positive
    protected static String toBinaryName(RelativePath relativePath);

    @Positive
    protected static String toBinaryName(Path relativePath);

    @Positive
    public static String getSimpleName(FileObject fo);

    @Positive
    public static class CannotCreateUriError extends Error {

    @Positive
        public CannotCreateUriError(String value, Throwable cause) {
    @Positive
        }
    @Positive
    }
    @Positive
}

// CFWR semantic augmentation - variant 0
