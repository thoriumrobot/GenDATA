/*
    @Positive
 * Copyright (c) 2000, 2021, Oracle and/or its affiliates. All rights reserved.
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
package java.beans;

    @Positive
import org.checkerframework.checker.interning.qual.UsesObjectEquals;
    @Positive
import org.checkerframework.framework.qual.AnnotatedFor;
    @Positive
import java.lang.reflect.AccessibleObject;
    @Positive
import java.lang.reflect.Array;
    @Positive
import java.lang.reflect.Constructor;
    @Positive
import java.lang.reflect.InvocationTargetException;
    @Positive
import java.lang.reflect.Method;
    @Positive
import java.security.AccessControlContext;
    @Positive
import java.security.AccessController;
    @Positive
import java.security.PrivilegedActionException;
    @Positive
import java.security.PrivilegedExceptionAction;
    @Positive
import com.sun.beans.finder.ClassFinder;
    @Positive
import com.sun.beans.finder.ConstructorFinder;
    @Positive
import com.sun.beans.finder.MethodFinder;
    @Positive
import sun.reflect.misc.MethodUtil;
    @Positive
import static sun.reflect.misc.ReflectUtil.checkPackageAccess;

    @Positive
@AnnotatedFor({ "interning" })
    @Positive
@UsesObjectEquals
    @Positive
public class Statement {

    @Positive
    @ConstructorProperties({ "target", "methodName", "arguments" })
    @Positive
    public Statement(Object target, String methodName, Object[] arguments) {
    @Positive
    }

    @Positive
    public Object getTarget();

    @Positive
    public String getMethodName();

    @Positive
    public Object[] getArguments();

    @Positive
    public void execute() throws Exception;

    @Positive
    @SuppressWarnings("removal")
    @Positive
    Object invoke() throws Exception;

    @Positive
    String instanceName(Object instance);

    @Positive
    public String toString();

    @Positive
    static Method getMethod(Class<?> type, String name, Class<?>... args);
    @Positive
}

// CFWR semantic augmentation - variant 0
