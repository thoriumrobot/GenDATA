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
package java.net;

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
import java.io.ObjectInputStream;
    @Positive
import java.io.ObjectOutputStream;
    @Positive
import java.io.ObjectStreamField;
    @Positive
import java.io.Serializable;
    @Positive
import java.net.InetAddress;
    @Positive
import java.security.AccessController;
    @Positive
import java.security.Permission;
    @Positive
import java.security.PermissionCollection;
    @Positive
import java.security.PrivilegedAction;
    @Positive
import java.security.Security;
    @Positive
import java.util.Collections;
    @Positive
import java.util.Enumeration;
    @Positive
import java.util.Map;
    @Positive
import java.util.StringJoiner;
    @Positive
import java.util.StringTokenizer;
    @Positive
import java.util.Vector;
    @Positive
import java.util.concurrent.ConcurrentHashMap;
    @Positive
import sun.net.util.IPAddressUtil;
    @Positive
import sun.net.PortConfig;
    @Positive
import sun.security.util.RegisteredDomain;
    @Positive
import sun.security.util.SecurityConstants;
    @Positive
import sun.security.util.Debug;

    @Positive
public final class SocketPermission extends Permission implements java.io.Serializable {

    @Positive
    private static class EphemeralRange {
    @Positive
    }

    @Positive
    public SocketPermission(String host, String action) {
    @Positive
    }

    @Positive
    void getCanonName() throws UnknownHostException;

    @Positive
    void getIP() throws UnknownHostException;

    @Positive
    @Override
    @Positive
    public boolean implies(Permission p);

    @Positive
    boolean impliesIgnoreMask(SocketPermission that);

    @Positive
    @Override
    @Positive
    @Pure
    @Positive
    @EnsuresNonNullIf(expression = "#1", result = true)
    @Positive
    public boolean equals(@Nullable Object obj);

    @Positive
    @Override
    @Positive
    public int hashCode();

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
}

    @Positive
final class SocketPermissionCollection extends PermissionCollection implements Serializable {

    @Positive
    public SocketPermissionCollection() {
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
