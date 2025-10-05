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
package sun.nio.fs;

import org.checkerframework.checker.nullness.qual.EnsuresNonNullIf;
import org.checkerframework.checker.nullness.qual.NonNull;
import org.checkerframework.checker.nullness.qual.Nullable;
import org.checkerframework.dataflow.qual.Pure;
import org.checkerframework.dataflow.qual.SideEffectFree;
import sun.nio.cs.UTF_8;
import jdk.internal.util.StaticProperty;
import java.nio.file.*;
import java.nio.file.attribute.*;
import java.nio.channels.*;
import java.util.*;
import java.io.IOException;
import java.security.AccessController;
import java.security.PrivilegedAction;

abstract class UnixFileStore extends FileStore {

    abstract UnixMountEntry findMountEntry() throws IOException;

    UnixPath file();

    long dev();

    UnixMountEntry entry();

    @Override
    public String name();

    @Override
    public String type();

    @Override
    public boolean isReadOnly();

    @Override
    public long getTotalSpace() throws IOException;

    @Override
    public long getUsableSpace() throws IOException;

    @Override
    public long getUnallocatedSpace() throws IOException;

    @Override
    public long getBlockSize() throws IOException;

    @Override
    public <V extends FileStoreAttributeView> V getFileStoreAttributeView(Class<V> view);

    @Override
    public Object getAttribute(String attribute) throws IOException;

    protected boolean isExtendedAttributesEnabled(UnixPath path);

    @Override
    public boolean supportsFileAttributeView(Class<? extends FileAttributeView> type);

    @Override
    public boolean supportsFileAttributeView(String name);

    @Override
    @Pure
    @EnsuresNonNullIf(expression = "#1", result = true)
    public boolean equals(@Nullable Object ob);

    @Override
    public int hashCode();

    @Override
    public String toString();

    @SuppressWarnings("removal")
    FeatureStatus checkIfFeaturePresent(String feature);
}
