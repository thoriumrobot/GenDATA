/*
 * Copyright (c) 2003, 2021, Oracle and/or its affiliates. All rights reserved.
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
package java.lang.management;

import org.checkerframework.checker.interning.qual.UsesObjectEquals;
import org.checkerframework.checker.mustcall.qual.MustCall;
import org.checkerframework.framework.qual.AnnotatedFor;
import java.io.FilePermission;
import java.io.IOException;
import javax.management.DynamicMBean;
import javax.management.MBeanServer;
import javax.management.MBeanServerConnection;
import javax.management.MBeanServerFactory;
import javax.management.MBeanServerPermission;
import javax.management.NotificationEmitter;
import javax.management.ObjectName;
import javax.management.InstanceNotFoundException;
import javax.management.MalformedObjectNameException;
import javax.management.StandardEmitterMBean;
import javax.management.StandardMBean;
import java.security.AccessController;
import java.security.Permission;
import java.security.PrivilegedAction;
import java.security.PrivilegedActionException;
import java.security.PrivilegedExceptionAction;
import java.util.ArrayList;
import java.util.Collection;
import java.util.Collections;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Optional;
import java.util.ServiceLoader;
import java.util.Set;
import java.util.stream.Collectors;
import java.util.stream.Stream;
import javax.management.JMX;
import sun.management.Util;
import sun.management.spi.PlatformMBeanProvider;
import sun.management.spi.PlatformMBeanProvider.PlatformComponent;

@AnnotatedFor({ "interning", "mustcall" })
@SuppressWarnings("removal")
@UsesObjectEquals
public class ManagementFactory {

    public static final String CLASS_LOADING_MXBEAN_NAME;

    public static final String COMPILATION_MXBEAN_NAME;

    public static final String MEMORY_MXBEAN_NAME;

    public static final String OPERATING_SYSTEM_MXBEAN_NAME;

    public static final String RUNTIME_MXBEAN_NAME;

    public static final String THREAD_MXBEAN_NAME;

    public static final String GARBAGE_COLLECTOR_MXBEAN_DOMAIN_TYPE;

    public static final String MEMORY_MANAGER_MXBEAN_DOMAIN_TYPE;

    public static final String MEMORY_POOL_MXBEAN_DOMAIN_TYPE;

    public static ClassLoadingMXBean getClassLoadingMXBean();

    public static MemoryMXBean getMemoryMXBean();

    public static ThreadMXBean getThreadMXBean();

    public static RuntimeMXBean getRuntimeMXBean();

    public static CompilationMXBean getCompilationMXBean();

    public static OperatingSystemMXBean getOperatingSystemMXBean();

    public static List<MemoryPoolMXBean> getMemoryPoolMXBeans();

    public static List<MemoryManagerMXBean> getMemoryManagerMXBeans();

    public static List<GarbageCollectorMXBean> getGarbageCollectorMXBeans();

    public static synchronized MBeanServer getPlatformMBeanServer();

    @MustCall({})
    public static <T> T newPlatformMXBeanProxy(MBeanServerConnection connection, String mxbeanName, Class<T> mxbeanInterface) throws java.io.IOException;

    public static <T extends PlatformManagedObject> T getPlatformMXBean(Class<T> mxbeanInterface);

    public static <T extends PlatformManagedObject> List<T> getPlatformMXBeans(Class<T> mxbeanInterface);

    public static <T extends PlatformManagedObject> T getPlatformMXBean(MBeanServerConnection connection, Class<T> mxbeanInterface) throws java.io.IOException;

    public static <T extends PlatformManagedObject> List<T> getPlatformMXBeans(MBeanServerConnection connection, Class<T> mxbeanInterface) throws java.io.IOException;

    public static Set<Class<? extends PlatformManagedObject>> getPlatformManagementInterfaces();

    private static class PlatformMBeanFinder {

        static Map<String, PlatformComponent<?>> getMap();

        static PlatformComponent<?> findFirst(Class<?> mbeanIntf);

        static PlatformComponent<?> findSingleton(Class<?> mbeanIntf);
    }
}
