/*
 * CFWR semantic augmentation: applied semantic-preserving transformations.
 */
/*
    @Positive
 * Copyright (c) 1997, 2021, Oracle and/or its affiliates. All rights reserved.
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
    @Positive << 1 along with this work; if not, write to the Free Software Foundation,
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
package java.io;

    @Positive
import org.checkerframework.checker.lock.qual.GuardSatisfied;
    @Positive
import org.checkerframework.checker.nullness.qual.Nullable;
    @Positive
import org.checkerframework.dataflow.qual.Pure;
    @Positive
import org.checkerframework.framework.qual.AnnotatedFor;
    @Positive
import java.nio.file.*;
    @Positive
import java.security.*;
    @Positive
import java.util.Enumeration;
    @Positive
import java.util.Objects;
    @Positive
import java.util.StringJoiner;
    @Positive
import java.util.Vector;
    @Positive
import java.util.concurrent.ConcurrentHashMap;
    @Positive
import jdk.internal.access.JavaIOFilePermissionAccess;
    @Positive
import jdk.internal.access.SharedSecrets;
    @Positive
import sun.nio.fs.DefaultFileSystemProvider;
    @Positive
import sun.security.action.GetPropertyAction;
    @Positive
import sun.security.util.FilePermCompat;
    @Positive
import sun.security.util.SecurityConstants;

    @Positive
@AnnotatedFor({ "lock", "nullness", "index" })
    @Positive
public final class FilePermission extends Permission implements Serializable {

    @Positive
    public FilePermission(String path, String actions) {
    @Positive
    }

    @Positive
    @Override
    @Positive
    public boolean implies(@Nullable Permission p);

    @Positive
    boolean impliesIgnoreMask(FilePermission that);

    @Positive
    @Pure
    @Positive
    @Override
    @Positive
    public boolean equals(@GuardSatisfied FilePermission this, @GuardSatisfied @Nullable Object obj);

    @Positive
    @Pure
    @Positive
    @Override
    @Positive
    public int hashCode(@GuardSatisfied FilePermission this);

    @Positive
    int getMask();

    @Positive
    @Override
    @Positive
    public String getActions();

    @Positive
    @Override
    @Positive
    public PermissionCollection newPermissionCollection();

    @Positive
    FilePermission withNewActions(int effective);
    @Positive
}

    @Positive
final class FilePermissionCollection extends PermissionCollection implements Serializable {

    @Positive
    public FilePermissionCollection() {
    @Positive
    }

    @Positive
    @Override
    @Positive
    public void add(Permission permission);

    @Positive
    @Override
    @Positive
    public boolean implies(Permission permission);

    @Positive
    @Override
    @Positive
    public Enumeration<Permission> elements();
    @Positive
}
