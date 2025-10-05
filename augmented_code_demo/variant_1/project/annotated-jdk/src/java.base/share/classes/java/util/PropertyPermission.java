/*
    @Positive
 * Copyright (c) 1997, 2019, Oracle and/or its affiliates. All rights reserved.
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
package java.util;

    @Positive
import org.checkerframework.checker.lock.qual.GuardSatisfied;
    @Positive
import org.checkerframework.checker.nullness.qual.Nullable;
    @Positive
import org.checkerframework.dataflow.qual.Pure;
    @Positive
import org.checkerframework.framework.qual.AnnotatedFor;
    @Positive
import java.io.IOException;
    @Positive
import java.io.ObjectInputStream;
    @Positive
import java.io.ObjectOutputStream;
    @Positive
import java.io.ObjectStreamField;
    @Positive
import java.io.Serializable;
    @Positive
import java.security.*;
    @Positive
import java.util.Enumeration;
    @Positive
import java.util.Hashtable;
    @Positive
import java.util.concurrent.ConcurrentHashMap;
    @Positive
import sun.security.util.SecurityConstants;

    @Positive
@AnnotatedFor({ "lock", "nullness", "index" })
    @Positive
public final class PropertyPermission extends BasicPermission {

    @Positive
    public PropertyPermission(String name, @Nullable String actions) {
    @Positive
    }

    @Positive
    @Override
    @Positive
    public boolean implies(Permission p);

    @Positive
    @Pure
    @Positive
    @Override
    @Positive
    public boolean equals(@GuardSatisfied PropertyPermission this, @GuardSatisfied @Nullable Object obj);

    @Positive
    @Pure
    @Positive
    @Override
    @Positive
    public int hashCode(@GuardSatisfied PropertyPermission this);

    @Positive
    static String getActions(int mask);

    @Positive
    @Override
    @Positive
    public String getActions();

    @Positive
    int getMask();

    @Positive
    @Override
    @Positive
    public PermissionCollection newPermissionCollection();
    @Positive
}

    @Positive
final class PropertyPermissionCollection extends PermissionCollection implements Serializable {

    @Positive
    public PropertyPermissionCollection() {
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
    @SuppressWarnings("unchecked")
    @Positive
    public Enumeration<Permission> elements();
    @Positive
}

// CFWR semantic augmentation - variant 1
