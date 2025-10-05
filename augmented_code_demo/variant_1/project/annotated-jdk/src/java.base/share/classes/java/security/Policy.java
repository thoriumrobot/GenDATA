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
import org.checkerframework.framework.qual.AnnotatedFor;
    @Positive
import java.util.Enumeration;
    @Positive
import java.util.WeakHashMap;
    @Positive
import java.util.Objects;
    @Positive
import sun.security.jca.GetInstance;
    @Positive
import sun.security.util.Debug;
    @Positive
import sun.security.util.SecurityConstants;

    @Positive
@AnnotatedFor({ "interning" })
    @Positive
@Deprecated()
    @Positive
@UsesObjectEquals
    @Positive
public abstract class Policy {

    @Positive
    public Policy() {
    @Positive
    }

    @Positive
    public static final PermissionCollection UNSUPPORTED_EMPTY_COLLECTION;

    @Positive
    private static class PolicyInfo {
    @Positive
    }

    @Positive
    static boolean isSet();

    @Positive
    public static Policy getPolicy();

    @Positive
    static Policy getPolicyNoCheck();

    @Positive
    public static void setPolicy(Policy p);

    @Positive
    @SuppressWarnings("removal")
    @Positive
    public static Policy getInstance(String type, Policy.Parameters params) throws NoSuchAlgorithmException;

    @Positive
    @SuppressWarnings("removal")
    @Positive
    public static Policy getInstance(String type, Policy.Parameters params, String provider) throws NoSuchProviderException, NoSuchAlgorithmException;

    @Positive
    @SuppressWarnings("removal")
    @Positive
    public static Policy getInstance(String type, Policy.Parameters params, Provider provider) throws NoSuchAlgorithmException;

    @Positive
    public Provider getProvider();

    @Positive
    public String getType();

    @Positive
    public Policy.Parameters getParameters();

    @Positive
    public PermissionCollection getPermissions(CodeSource codesource);

    @Positive
    public PermissionCollection getPermissions(ProtectionDomain domain);

    @Positive
    public boolean implies(ProtectionDomain domain, Permission permission);

    @Positive
    public void refresh();

    @Positive
    private static class PolicyDelegate extends Policy {

    @Positive
        @Override
    @Positive
        public String getType();

    @Positive
        @Override
    @Positive
        public Policy.Parameters getParameters();

    @Positive
        @Override
    @Positive
        public Provider getProvider();

    @Positive
        @Override
    @Positive
        public PermissionCollection getPermissions(CodeSource codesource);

    @Positive
        @Override
    @Positive
        public PermissionCollection getPermissions(ProtectionDomain domain);

    @Positive
        @Override
    @Positive
        public boolean implies(ProtectionDomain domain, Permission perm);

    @Positive
        @Override
    @Positive
        public void refresh();
    @Positive
    }

    @Positive
    @Deprecated()
    @Positive
    public static interface Parameters {
    @Positive
    }

    @Positive
    private static class UnsupportedEmptyCollection extends PermissionCollection {

    @Positive
        public UnsupportedEmptyCollection() {
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
    @Positive
}

// CFWR semantic augmentation - variant 1
