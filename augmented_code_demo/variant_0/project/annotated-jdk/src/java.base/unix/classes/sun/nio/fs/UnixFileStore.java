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
package sun.nio.fs;

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
import sun.nio.cs.UTF_8;
    @Positive
import jdk.internal.util.StaticProperty;
    @Positive
import java.nio.file.*;
    @Positive
import java.nio.file.attribute.*;
    @Positive
import java.nio.channels.*;
    @Positive
import java.util.*;
    @Positive
import java.io.IOException;
    @Positive
import java.security.AccessController;
    @Positive
import java.security.PrivilegedAction;

    @Positive
abstract class UnixFileStore extends FileStore {

    @Positive
    abstract UnixMountEntry findMountEntry() throws IOException;

    @Positive
    UnixPath file();

    @Positive
    long dev();

    @Positive
    UnixMountEntry entry();

    @Positive
    @Override
    @Positive
    public String name();

    @Positive
    @Override
    @Positive
    public String type();

    @Positive
    @Override
    @Positive
    public boolean isReadOnly();

    @Positive
    @Override
    @Positive
    public long getTotalSpace() throws IOException;

    @Positive
    @Override
    @Positive
    public long getUsableSpace() throws IOException;

    @Positive
    @Override
    @Positive
    public long getUnallocatedSpace() throws IOException;

    @Positive
    @Override
    @Positive
    public long getBlockSize() throws IOException;

    @Positive
    @Override
    @Positive
    public <V extends FileStoreAttributeView> V getFileStoreAttributeView(Class<V> view);

    @Positive
    @Override
    @Positive
    public Object getAttribute(String attribute) throws IOException;

    @Positive
    protected boolean isExtendedAttributesEnabled(UnixPath path);

    @Positive
    @Override
    @Positive
    public boolean supportsFileAttributeView(Class<? extends FileAttributeView> type);

    @Positive
    @Override
    @Positive
    public boolean supportsFileAttributeView(String name);

    @Positive
    @Override
    @Positive
    @Pure
    @Positive
    @EnsuresNonNullIf(expression = "#1", result = true)
    @Positive
    public boolean equals(@Nullable Object ob);

    @Positive
    @Override
    @Positive
    public int hashCode();

    @Positive
    @Override
    @Positive
    public String toString();

    @Positive
    @SuppressWarnings("removal")
    @Positive
    FeatureStatus checkIfFeaturePresent(String feature);
    @Positive
}

// CFWR semantic augmentation - variant 0
