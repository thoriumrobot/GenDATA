/*
    @Positive
 * Copyright (c) 1996, 2021, Oracle and/or its affiliates. All rights reserved.
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
package java.sql;

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
import org.checkerframework.checker.interning.qual.UsesObjectEquals;
    @Positive
import org.checkerframework.framework.qual.AnnotatedFor;
    @Positive
import java.util.ArrayList;
    @Positive
import java.util.Collections;
    @Positive
import java.util.Enumeration;
    @Positive
import java.util.Iterator;
    @Positive
import java.util.List;
    @Positive
import java.util.ServiceLoader;
    @Positive
import java.security.AccessController;
    @Positive
import java.security.PrivilegedAction;
    @Positive
import java.util.concurrent.CopyOnWriteArrayList;
    @Positive
import java.util.stream.Stream;
    @Positive
import jdk.internal.reflect.CallerSensitive;
    @Positive
import jdk.internal.reflect.Reflection;

    @Positive
@AnnotatedFor({ "interning" })
    @Positive
@UsesObjectEquals
    @Positive
public class DriverManager {

    @Positive
    public static java.io.PrintWriter getLogWriter();

    @Positive
    public static void setLogWriter(java.io.PrintWriter out);

    @Positive
    @CallerSensitive
    @Positive
    public static Connection getConnection(String url, java.util.Properties info) throws SQLException;

    @Positive
    @CallerSensitive
    @Positive
    public static Connection getConnection(String url, String user, String password) throws SQLException;

    @Positive
    @CallerSensitive
    @Positive
    public static Connection getConnection(String url) throws SQLException;

    @Positive
    @CallerSensitive
    @Positive
    public static Driver getDriver(String url) throws SQLException;

    @Positive
    public static void registerDriver(java.sql.Driver driver) throws SQLException;

    @Positive
    public static void registerDriver(java.sql.Driver driver, DriverAction da) throws SQLException;

    @Positive
    @CallerSensitive
    @Positive
    public static void deregisterDriver(Driver driver) throws SQLException;

    @Positive
    @CallerSensitive
    @Positive
    public static Enumeration<Driver> getDrivers();

    @Positive
    @CallerSensitive
    @Positive
    public static Stream<Driver> drivers();

    @Positive
    public static void setLoginTimeout(int seconds);

    @Positive
    public static int getLoginTimeout();

    @Positive
    @Deprecated()
    @Positive
    public static void setLogStream(java.io.PrintStream out);

    @Positive
    @Deprecated()
    @Positive
    public static java.io.PrintStream getLogStream();

    @Positive
    public static void println(String message);
    @Positive
}

    @Positive
class DriverInfo {

    @Positive
    @Override
    @Positive
    @Pure
    @Positive
    @EnsuresNonNullIf(expression = "#1", result = true)
    @Positive
    public boolean equals(@Nullable Object other);

    @Positive
    @Override
    @Positive
    public int hashCode();

    @Positive
    @Override
    @Positive
    public String toString();

    @Positive
    DriverAction action();
    @Positive
}

// CFWR semantic augmentation - variant 0
