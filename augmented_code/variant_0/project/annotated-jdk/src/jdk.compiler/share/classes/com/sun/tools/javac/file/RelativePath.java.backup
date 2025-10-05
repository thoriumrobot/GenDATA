/*
 * Copyright (c) 2008, 2021, Oracle and/or its affiliates. All rights reserved.
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
import java.nio.file.FileSystem;
import java.nio.file.InvalidPathException;
import java.nio.file.Path;
import java.util.zip.ZipEntry;
import java.util.zip.ZipFile;
import javax.tools.JavaFileObject;

public abstract class RelativePath implements Comparable<RelativePath> {

    protected RelativePath(String p) {
    }

    public abstract RelativeDirectory dirname();

    public abstract String basename();

    public Path resolveAgainst(Path directory) throws InvalidPathException;

    public Path resolveAgainst(FileSystem fs) throws InvalidPathException;

    @Override
    public int compareTo(RelativePath other);

    @Override
    @Pure
    @EnsuresNonNullIf(expression = "#1", result = true)
    public boolean equals(@Nullable Object other);

    @Override
    public int hashCode();

    @Override
    public String toString();

    public String getPath();

    protected final String path;

    public static class RelativeDirectory extends RelativePath {

        static RelativeDirectory forPackage(CharSequence packageName);

        public RelativeDirectory(String p) {
        }

        public RelativeDirectory(RelativeDirectory d, String p) {
        }

        @Override
        public RelativeDirectory dirname();

        @Override
        public String basename();

        @Pure
        boolean contains(RelativePath other);

        @Override
        public String toString();
    }

    public static class RelativeFile extends RelativePath {

        static RelativeFile forClass(CharSequence className, JavaFileObject.Kind kind);

        public RelativeFile(String p) {
        }

        public RelativeFile(RelativeDirectory d, String p) {
        }

        @Override
        public RelativeDirectory dirname();

        @Override
        public String basename();

        ZipEntry getZipEntry(ZipFile zip);

        @Override
        public String toString();
    }
}
