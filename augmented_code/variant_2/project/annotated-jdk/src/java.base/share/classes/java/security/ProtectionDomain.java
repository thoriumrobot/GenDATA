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
package java.security;

    @Positive
import org.checkerframework.checker.interning.qual.UsesObjectEquals;
    @Positive
import org.checkerframework.checker.nullness.qual.Nullable;
    @Positive
import org.checkerframework.dataflow.qual.Deterministic;
    @Positive
import org.checkerframework.framework.qual.AnnotatedFor;
    @Positive
import java.util.ArrayList;
    @Positive
import java.util.Collections;
    @Positive
import java.util.Enumeration;
    @Positive
import java.util.List;
    @Positive
import java.util.Map;
    @Positive
import java.util.Objects;
    @Positive
import java.util.WeakHashMap;
    @Positive
import jdk.internal.access.JavaSecurityAccess;
    @Positive
import jdk.internal.access.SharedSecrets;
    @Positive
import sun.security.action.GetPropertyAction;
    @Positive
import sun.security.provider.PolicyFile;
    @Positive
import sun.security.util.Debug;
    @Positive
import sun.security.util.FilePermCompat;
    @Positive
import sun.security.util.SecurityConstants;

    @Positive
@AnnotatedFor({ "interning", "nullness" })
    @Positive
@UsesObjectEquals
    @Positive
public class ProtectionDomain {

    @Positive
    private static class JavaSecurityAccessImpl implements JavaSecurityAccess {

    @Positive
        @SuppressWarnings("removal")
    @Positive
        @Override
    @Positive
        public <T> T doIntersectionPrivilege(PrivilegedAction<T> action, final AccessControlContext stack, final AccessControlContext context);

    @Positive
        @SuppressWarnings("removal")
    @Positive
        @Override
    @Positive
        public <T> T doIntersectionPrivilege(PrivilegedAction<T> action, AccessControlContext context);

    @Positive
        @Override
    @Positive
        public ProtectionDomain[] getProtectDomains(@SuppressWarnings("removal") AccessControlContext context);

    @Positive
        @Override
    @Positive
        public ProtectionDomainCache getProtectionDomainCache();
    @Positive
    }

    @Positive
    public ProtectionDomain(@Nullable CodeSource codesource, @Nullable PermissionCollection permissions) {
    @Positive
    }

    @Positive
    public ProtectionDomain(@Nullable CodeSource codesource, @Nullable PermissionCollection permissions, @Nullable ClassLoader classloader, Principal[] principals) {
    @Positive
    }

    @Positive
    @Deterministic
    @Positive
    @Nullable
    @Positive
    public final CodeSource getCodeSource();

    @Positive
    @Nullable
    @Positive
    public final ClassLoader getClassLoader();

    @Positive
    public final Principal[] getPrincipals();

    @Positive
    @Nullable
    @Positive
    public final PermissionCollection getPermissions();

    @Positive
    public final boolean staticPermissionsOnly();

    @Positive
    @SuppressWarnings("removal")
    @Positive
    public boolean implies(Permission perm);

    @Positive
    boolean impliesWithAltFilePerm(Permission perm);

    @Positive
    @Override
    @Positive
    public String toString();

    @Positive
    private static class DebugHolder {
    @Positive
    }

    @Positive
    static final class Key {
    @Positive
    }
    @Positive
}
