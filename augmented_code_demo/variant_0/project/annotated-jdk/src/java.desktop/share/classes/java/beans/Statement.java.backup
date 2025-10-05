/*
 * Copyright (c) 2000, 2021, Oracle and/or its affiliates. All rights reserved.
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
package java.beans;

import org.checkerframework.checker.interning.qual.UsesObjectEquals;
import org.checkerframework.framework.qual.AnnotatedFor;
import java.lang.reflect.AccessibleObject;
import java.lang.reflect.Array;
import java.lang.reflect.Constructor;
import java.lang.reflect.InvocationTargetException;
import java.lang.reflect.Method;
import java.security.AccessControlContext;
import java.security.AccessController;
import java.security.PrivilegedActionException;
import java.security.PrivilegedExceptionAction;
import com.sun.beans.finder.ClassFinder;
import com.sun.beans.finder.ConstructorFinder;
import com.sun.beans.finder.MethodFinder;
import sun.reflect.misc.MethodUtil;
import static sun.reflect.misc.ReflectUtil.checkPackageAccess;

@AnnotatedFor({ "interning" })
@UsesObjectEquals
public class Statement {

    @ConstructorProperties({ "target", "methodName", "arguments" })
    public Statement(Object target, String methodName, Object[] arguments) {
    }

    public Object getTarget();

    public String getMethodName();

    public Object[] getArguments();

    public void execute() throws Exception;

    @SuppressWarnings("removal")
    Object invoke() throws Exception;

    String instanceName(Object instance);

    public String toString();

    static Method getMethod(Class<?> type, String name, Class<?>... args);
}
