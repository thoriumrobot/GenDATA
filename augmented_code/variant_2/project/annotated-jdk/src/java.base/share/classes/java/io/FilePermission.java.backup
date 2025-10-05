/*
 * Copyright (c) 1997, 2021, Oracle and/or its affiliates. All rights reserved.
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
package java.io;

import org.checkerframework.checker.lock.qual.GuardSatisfied;
import org.checkerframework.checker.nullness.qual.Nullable;
import org.checkerframework.dataflow.qual.Pure;
import org.checkerframework.framework.qual.AnnotatedFor;
import java.nio.file.*;
import java.security.*;
import java.util.Enumeration;
import java.util.Objects;
import java.util.StringJoiner;
import java.util.Vector;
import java.util.concurrent.ConcurrentHashMap;
import jdk.internal.access.JavaIOFilePermissionAccess;
import jdk.internal.access.SharedSecrets;
import sun.nio.fs.DefaultFileSystemProvider;
import sun.security.action.GetPropertyAction;
import sun.security.util.FilePermCompat;
import sun.security.util.SecurityConstants;

@AnnotatedFor({ "lock", "nullness", "index" })
public final class FilePermission extends Permission implements Serializable {

    public FilePermission(String path, String actions) {
    }

    @Override
    public boolean implies(@Nullable Permission p);

    boolean impliesIgnoreMask(FilePermission that);

    @Pure
    @Override
    public boolean equals(@GuardSatisfied FilePermission this, @GuardSatisfied @Nullable Object obj);

    @Pure
    @Override
    public int hashCode(@GuardSatisfied FilePermission this);

    int getMask();

    @Override
    public String getActions();

    @Override
    public PermissionCollection newPermissionCollection();

    FilePermission withNewActions(int effective);
}

final class FilePermissionCollection extends PermissionCollection implements Serializable {

    public FilePermissionCollection() {
    }

    @Override
    public void add(Permission permission);

    @Override
    public boolean implies(Permission permission);

    @Override
    public Enumeration<Permission> elements();
}
