/*
 * Copyright (c) 1996, 2021, Oracle and/or its affiliates. All rights reserved.
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
package java.sql;

import org.checkerframework.checker.nullness.qual.EnsuresNonNullIf;
import org.checkerframework.checker.nullness.qual.NonNull;
import org.checkerframework.checker.nullness.qual.Nullable;
import org.checkerframework.dataflow.qual.Pure;
import org.checkerframework.dataflow.qual.SideEffectFree;
import org.checkerframework.checker.interning.qual.UsesObjectEquals;
import org.checkerframework.framework.qual.AnnotatedFor;
import java.util.ArrayList;
import java.util.Collections;
import java.util.Enumeration;
import java.util.Iterator;
import java.util.List;
import java.util.ServiceLoader;
import java.security.AccessController;
import java.security.PrivilegedAction;
import java.util.concurrent.CopyOnWriteArrayList;
import java.util.stream.Stream;
import jdk.internal.reflect.CallerSensitive;
import jdk.internal.reflect.Reflection;

@AnnotatedFor({ "interning" })
@UsesObjectEquals
public class DriverManager {

    public static java.io.PrintWriter getLogWriter();

    public static void setLogWriter(java.io.PrintWriter out);

    @CallerSensitive
    public static Connection getConnection(String url, java.util.Properties info) throws SQLException;

    @CallerSensitive
    public static Connection getConnection(String url, String user, String password) throws SQLException;

    @CallerSensitive
    public static Connection getConnection(String url) throws SQLException;

    @CallerSensitive
    public static Driver getDriver(String url) throws SQLException;

    public static void registerDriver(java.sql.Driver driver) throws SQLException;

    public static void registerDriver(java.sql.Driver driver, DriverAction da) throws SQLException;

    @CallerSensitive
    public static void deregisterDriver(Driver driver) throws SQLException;

    @CallerSensitive
    public static Enumeration<Driver> getDrivers();

    @CallerSensitive
    public static Stream<Driver> drivers();

    public static void setLoginTimeout(int seconds);

    public static int getLoginTimeout();

    @Deprecated()
    public static void setLogStream(java.io.PrintStream out);

    @Deprecated()
    public static java.io.PrintStream getLogStream();

    public static void println(String message);
}

class DriverInfo {

    @Override
    @Pure
    @EnsuresNonNullIf(expression = "#1", result = true)
    public boolean equals(@Nullable Object other);

    @Override
    public int hashCode();

    @Override
    public String toString();

    DriverAction action();
}
