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
package java.security;

import org.checkerframework.checker.interning.qual.UsesObjectEquals;
import org.checkerframework.checker.nullness.qual.Nullable;
import org.checkerframework.dataflow.qual.Deterministic;
import org.checkerframework.framework.qual.AnnotatedFor;
import java.util.ArrayList;
import java.util.Collections;
import java.util.Enumeration;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.WeakHashMap;
import jdk.internal.access.JavaSecurityAccess;
import jdk.internal.access.SharedSecrets;
import sun.security.action.GetPropertyAction;
import sun.security.provider.PolicyFile;
import sun.security.util.Debug;
import sun.security.util.FilePermCompat;
import sun.security.util.SecurityConstants;

@AnnotatedFor({ "interning", "nullness" })
@UsesObjectEquals
public class ProtectionDomain {

    private static class JavaSecurityAccessImpl implements JavaSecurityAccess {

        @SuppressWarnings("removal")
        @Override
        public <T> T doIntersectionPrivilege(PrivilegedAction<T> action, final AccessControlContext stack, final AccessControlContext context);

        @SuppressWarnings("removal")
        @Override
        public <T> T doIntersectionPrivilege(PrivilegedAction<T> action, AccessControlContext context);

        @Override
        public ProtectionDomain[] getProtectDomains(@SuppressWarnings("removal") AccessControlContext context);

        @Override
        public ProtectionDomainCache getProtectionDomainCache();
    }

    public ProtectionDomain(@Nullable CodeSource codesource, @Nullable PermissionCollection permissions) {
    }

    public ProtectionDomain(@Nullable CodeSource codesource, @Nullable PermissionCollection permissions, @Nullable ClassLoader classloader, Principal[] principals) {
    }

    @Deterministic
    @Nullable
    public final CodeSource getCodeSource();

    @Nullable
    public final ClassLoader getClassLoader();

    public final Principal[] getPrincipals();

    @Nullable
    public final PermissionCollection getPermissions();

    public final boolean staticPermissionsOnly();

    @SuppressWarnings("removal")
    public boolean implies(Permission perm);

    boolean impliesWithAltFilePerm(Permission perm);

    @Override
    public String toString();

    private static class DebugHolder {
    }

    static final class Key {
    }
}
