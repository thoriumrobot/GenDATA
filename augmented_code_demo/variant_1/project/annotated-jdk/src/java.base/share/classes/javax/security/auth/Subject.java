/*
    @Positive
 * Copyright (c) 1998, 2021, Oracle and/or its affiliates. All rights reserved.
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
package javax.security.auth;

    @Positive
import org.checkerframework.checker.nonempty.qual.EnsuresNonEmpty;
    @Positive
import org.checkerframework.checker.nonempty.qual.EnsuresNonEmptyIf;
    @Positive
import org.checkerframework.checker.nonempty.qual.NonEmpty;
    @Positive
import org.checkerframework.checker.nullness.qual.EnsuresNonNullIf;
    @Positive
import org.checkerframework.checker.nullness.qual.NonNull;
    @Positive
import org.checkerframework.checker.nullness.qual.Nullable;
    @Positive
import org.checkerframework.checker.signedness.qual.UnknownSignedness;
    @Positive
import org.checkerframework.dataflow.qual.Pure;
    @Positive
import org.checkerframework.dataflow.qual.SideEffectFree;
    @Positive
import org.checkerframework.dataflow.qual.SideEffectsOnly;
    @Positive
import java.util.*;
    @Positive
import java.io.*;
    @Positive
import java.lang.reflect.*;
    @Positive
import java.text.MessageFormat;
    @Positive
import java.security.AccessController;
    @Positive
import java.security.AccessControlContext;
    @Positive
import java.security.DomainCombiner;
    @Positive
import java.security.Permission;
    @Positive
import java.security.PermissionCollection;
    @Positive
import java.security.Principal;
    @Positive
import java.security.PrivilegedAction;
    @Positive
import java.security.PrivilegedExceptionAction;
    @Positive
import java.security.PrivilegedActionException;
    @Positive
import java.security.ProtectionDomain;
    @Positive
import sun.security.util.ResourcesMgr;

    @Positive
public final class Subject implements java.io.Serializable {

    @Positive
    public Subject() {
    @Positive
    }

    @Positive
    public Subject(boolean readOnly, Set<? extends Principal> principals, Set<?> pubCredentials, Set<?> privCredentials) {
    @Positive
    }

    @Positive
    public void setReadOnly();

    @Positive
    public boolean isReadOnly();

    @Positive
    @SuppressWarnings("removal")
    @Positive
    @Deprecated()
    @Positive
    public static Subject getSubject(final AccessControlContext acc);

    @Positive
    @SuppressWarnings("removal")
    @Positive
    public static <T> T doAs(final Subject subject, final java.security.PrivilegedAction<T> action);

    @Positive
    @SuppressWarnings("removal")
    @Positive
    public static <T> T doAs(final Subject subject, final java.security.PrivilegedExceptionAction<T> action) throws java.security.PrivilegedActionException;

    @Positive
    @SuppressWarnings("removal")
    @Positive
    @Deprecated()
    @Positive
    public static <T> T doAsPrivileged(final Subject subject, final java.security.PrivilegedAction<T> action, final java.security.AccessControlContext acc);

    @Positive
    @SuppressWarnings("removal")
    @Positive
    @Deprecated()
    @Positive
    public static <T> T doAsPrivileged(final Subject subject, final java.security.PrivilegedExceptionAction<T> action, final java.security.AccessControlContext acc) throws java.security.PrivilegedActionException;

    @Positive
    public Set<Principal> getPrincipals();

    @Positive
    public <T extends Principal> Set<T> getPrincipals(Class<T> c);

    @Positive
    public Set<Object> getPublicCredentials();

    @Positive
    public Set<Object> getPrivateCredentials();

    @Positive
    public <T> Set<T> getPublicCredentials(Class<T> c);

    @Positive
    public <T> Set<T> getPrivateCredentials(Class<T> c);

    @Positive
    @Override
    @Positive
    @Pure
    @Positive
    @EnsuresNonNullIf(expression = "#1", result = true)
    @Positive
    public boolean equals(@Nullable Object o);

    @Positive
    @Override
    @Positive
    public String toString();

    @Positive
    String toString(boolean includePrivateCredentials);

    @Positive
    @Override
    @Positive
    public int hashCode();

    @Positive
    private static class SecureSet<E> implements Set<E>, java.io.Serializable {

    @Positive
        public int size();

    @Positive
        public Iterator<E> iterator();

    @Positive
        @EnsuresNonEmpty("this")
    @Positive
        public boolean add(E o);

    @Positive
        @SuppressWarnings("removal")
    @Positive
        public boolean remove(@UnknownSignedness Object o);

    @Positive
        @SuppressWarnings("removal")
    @Positive
        @Pure
    @Positive
        @EnsuresNonEmptyIf(result = true, expression = "this")
    @Positive
        public boolean contains(@UnknownSignedness Object o);

    @Positive
        public boolean addAll(Collection<? extends E> c);

    @Positive
        @SuppressWarnings("removal")
    @Positive
        public boolean removeAll(Collection<? extends @UnknownSignedness Object> c);

    @Positive
        @Pure
    @Positive
        public boolean containsAll(Collection<? extends @UnknownSignedness Object> c);

    @Positive
        @SuppressWarnings("removal")
    @Positive
        public boolean retainAll(Collection<? extends @UnknownSignedness Object> c);

    @Positive
        @SuppressWarnings("removal")
    @Positive
        public void clear();

    @Positive
        @EnsuresNonEmptyIf(result = false, expression = "this")
    @Positive
        public boolean isEmpty();

    @Positive
        public Object[] toArray();

    @Positive
        public <T> T[] toArray(T[] a);

    @Positive
        public boolean equals(Object o);

    @Positive
        public int hashCode();
    @Positive
    }

    @Positive
    private class ClassSet<T> extends AbstractSet<T> {

    @Positive
        @Override
    @Positive
        public int size();

    @Positive
        @Override
    @Positive
        public Iterator<T> iterator();

    @Positive
        @Override
    @Positive
        @EnsuresNonEmpty("this")
    @Positive
        public boolean add(T o);
    @Positive
    }

    @Positive
    static final class AuthPermissionHolder {
    @Positive
    }
    @Positive
}

// CFWR semantic augmentation - variant 1
