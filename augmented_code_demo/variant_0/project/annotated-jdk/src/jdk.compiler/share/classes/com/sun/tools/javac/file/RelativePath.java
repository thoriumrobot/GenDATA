/*
    @Positive
 * Copyright (c) 2008, 2021, Oracle and/or its affiliates. All rights reserved.
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
import java.nio.file.FileSystem;
    @Positive
import java.nio.file.InvalidPathException;
    @Positive
import java.nio.file.Path;
    @Positive
import java.util.zip.ZipEntry;
    @Positive
import java.util.zip.ZipFile;
    @Positive
import javax.tools.JavaFileObject;

    @Positive
public abstract class RelativePath implements Comparable<RelativePath> {

    @Positive
    protected RelativePath(String p) {
    @Positive
    }

    @Positive
    public abstract RelativeDirectory dirname();

    @Positive
    public abstract String basename();

    @Positive
    public Path resolveAgainst(Path directory) throws InvalidPathException;

    @Positive
    public Path resolveAgainst(FileSystem fs) throws InvalidPathException;

    @Positive
    @Override
    @Positive
    public int compareTo(RelativePath other);

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
    public String getPath();

    @Positive
    protected final String path;

    @Positive
    public static class RelativeDirectory extends RelativePath {

    @Positive
        static RelativeDirectory forPackage(CharSequence packageName);

    @Positive
        public RelativeDirectory(String p) {
    @Positive
        }

    @Positive
        public RelativeDirectory(RelativeDirectory d, String p) {
    @Positive
        }

    @Positive
        @Override
    @Positive
        public RelativeDirectory dirname();

    @Positive
        @Override
    @Positive
        public String basename();

    @Positive
        @Pure
    @Positive
        boolean contains(RelativePath other);

    @Positive
        @Override
    @Positive
        public String toString();
    @Positive
    }

    @Positive
    public static class RelativeFile extends RelativePath {

    @Positive
        static RelativeFile forClass(CharSequence className, JavaFileObject.Kind kind);

    @Positive
        public RelativeFile(String p) {
    @Positive
        }

    @Positive
        public RelativeFile(RelativeDirectory d, String p) {
    @Positive
        }

    @Positive
        @Override
    @Positive
        public RelativeDirectory dirname();

    @Positive
        @Override
    @Positive
        public String basename();

    @Positive
        ZipEntry getZipEntry(ZipFile zip);

    @Positive
        @Override
    @Positive
        public String toString();
    @Positive
    }
    @Positive
}

// CFWR semantic augmentation - variant 0
