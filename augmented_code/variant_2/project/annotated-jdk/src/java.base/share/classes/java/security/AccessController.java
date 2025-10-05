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
import org.checkerframework.framework.qual.AnnotatedFor;
    @Positive
import java.lang.annotation.ElementType;
    @Positive
import java.lang.annotation.Retention;
    @Positive
import java.lang.annotation.RetentionPolicy;
    @Positive
import java.lang.annotation.Target;
    @Positive
import java.lang.ref.Reference;
    @Positive
import jdk.internal.vm.annotation.Hidden;
    @Positive
import sun.security.util.Debug;
    @Positive
import sun.security.util.SecurityConstants;
    @Positive
import jdk.internal.reflect.CallerSensitive;
    @Positive
import jdk.internal.reflect.Reflection;
    @Positive
import jdk.internal.vm.annotation.DontInline;
    @Positive
import jdk.internal.vm.annotation.ForceInline;
    @Positive
import jdk.internal.vm.annotation.ReservedStackAccess;

    @Positive
@AnnotatedFor({ "interning" })
    @Positive
@Deprecated()
    @Positive
@UsesObjectEquals
    @Positive
public final class AccessController {

    @Positive
    @CallerSensitive
    @Positive
    public static <T> T doPrivileged(PrivilegedAction<T> action);

    @Positive
    @CallerSensitive
    @Positive
    public static <T> T doPrivilegedWithCombiner(PrivilegedAction<T> action);

    @Positive
    @CallerSensitive
    @Positive
    public static <T> T doPrivileged(PrivilegedAction<T> action, @SuppressWarnings("removal") AccessControlContext context);

    @Positive
    @CallerSensitive
    @Positive
    public static <T> T doPrivileged(PrivilegedAction<T> action, @SuppressWarnings("removal") AccessControlContext context, Permission... perms);

    @Positive
    @CallerSensitive
    @Positive
    public static <T> T doPrivilegedWithCombiner(PrivilegedAction<T> action, @SuppressWarnings("removal") AccessControlContext context, Permission... perms);

    @Positive
    @CallerSensitive
    @Positive
    public static <T> T doPrivileged(PrivilegedExceptionAction<T> action) throws PrivilegedActionException;

    @Positive
    @CallerSensitive
    @Positive
    public static <T> T doPrivilegedWithCombiner(PrivilegedExceptionAction<T> action) throws PrivilegedActionException;

    @Positive
    private static class AccHolder {
    @Positive
    }

    @Positive
    @CallerSensitive
    @Positive
    public static <T> T doPrivileged(PrivilegedExceptionAction<T> action, @SuppressWarnings("removal") AccessControlContext context) throws PrivilegedActionException;

    @Positive
    @CallerSensitive
    @Positive
    public static <T> T doPrivileged(PrivilegedExceptionAction<T> action, @SuppressWarnings("removal") AccessControlContext context, Permission... perms) throws PrivilegedActionException;

    @Positive
    @CallerSensitive
    @Positive
    public static <T> T doPrivilegedWithCombiner(PrivilegedExceptionAction<T> action, @SuppressWarnings("removal") AccessControlContext context, Permission... perms) throws PrivilegedActionException;

    @Positive
    @SuppressWarnings("removal")
    @Positive
    static native AccessControlContext getInheritedAccessControlContext();

    @Positive
    @SuppressWarnings("removal")
    @Positive
    public static AccessControlContext getContext();

    @Positive
    @SuppressWarnings("removal")
    @Positive
    public static void checkPermission(Permission perm) throws AccessControlException;
    @Positive
}
