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
package java.net;

import org.checkerframework.checker.nullness.qual.EnsuresNonNullIf;
import org.checkerframework.checker.nullness.qual.NonNull;
import org.checkerframework.checker.nullness.qual.Nullable;
import org.checkerframework.dataflow.qual.Pure;
import org.checkerframework.dataflow.qual.SideEffectFree;
import java.io.IOException;
import java.io.ObjectInputStream;
import java.io.ObjectOutputStream;
import java.io.ObjectStreamField;
import java.io.Serializable;
import java.net.InetAddress;
import java.security.AccessController;
import java.security.Permission;
import java.security.PermissionCollection;
import java.security.PrivilegedAction;
import java.security.Security;
import java.util.Collections;
import java.util.Enumeration;
import java.util.Map;
import java.util.StringJoiner;
import java.util.StringTokenizer;
import java.util.Vector;
import java.util.concurrent.ConcurrentHashMap;
import sun.net.util.IPAddressUtil;
import sun.net.PortConfig;
import sun.security.util.RegisteredDomain;
import sun.security.util.SecurityConstants;
import sun.security.util.Debug;

public final class SocketPermission extends Permission implements java.io.Serializable {

    private static class EphemeralRange {
    }

    public SocketPermission(String host, String action) {
    }

    void getCanonName() throws UnknownHostException;

    void getIP() throws UnknownHostException;

    @Override
    public boolean implies(Permission p);

    boolean impliesIgnoreMask(SocketPermission that);

    @Override
    @Pure
    @EnsuresNonNullIf(expression = "#1", result = true)
    public boolean equals(@Nullable Object obj);

    @Override
    public int hashCode();

    int getMask();

    @Override
    public String getActions();

    @Override
    public PermissionCollection newPermissionCollection();
}

final class SocketPermissionCollection extends PermissionCollection implements Serializable {

    public SocketPermissionCollection() {
    }

    @Override
    public void add(Permission permission);

    @Override
    public boolean implies(Permission permission);

    @Override
    @SuppressWarnings("unchecked")
    public Enumeration<Permission> elements();
}
