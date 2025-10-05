/*
 * Copyright (c) 1997, 2019, Oracle and/or its affiliates. All rights reserved.
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
package java.util;

import org.checkerframework.checker.lock.qual.GuardSatisfied;
import org.checkerframework.checker.nullness.qual.Nullable;
import org.checkerframework.dataflow.qual.Pure;
import org.checkerframework.framework.qual.AnnotatedFor;
import java.io.IOException;
import java.io.ObjectInputStream;
import java.io.ObjectOutputStream;
import java.io.ObjectStreamField;
import java.io.Serializable;
import java.security.*;
import java.util.Enumeration;
import java.util.Hashtable;
import java.util.concurrent.ConcurrentHashMap;
import sun.security.util.SecurityConstants;

@AnnotatedFor({ "lock", "nullness", "index" })
public final class PropertyPermission extends BasicPermission {

    public PropertyPermission(String name, @Nullable String actions) {
    }

    @Override
    public boolean implies(Permission p);

    @Pure
    @Override
    public boolean equals(@GuardSatisfied PropertyPermission this, @GuardSatisfied @Nullable Object obj);

    @Pure
    @Override
    public int hashCode(@GuardSatisfied PropertyPermission this);

    static String getActions(int mask);

    @Override
    public String getActions();

    int getMask();

    @Override
    public PermissionCollection newPermissionCollection();
}

final class PropertyPermissionCollection extends PermissionCollection implements Serializable {

    public PropertyPermissionCollection() {
    }

    @Override
    public void add(Permission permission);

    @Override
    public boolean implies(Permission permission);

    @Override
    @SuppressWarnings("unchecked")
    public Enumeration<Permission> elements();
}
