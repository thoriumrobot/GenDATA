/*
    @Positive
 * Copyright (c) 2003, 2021, Oracle and/or its affiliates. All rights reserved.
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
package java.lang.management;

    @Positive
import org.checkerframework.checker.interning.qual.UsesObjectEquals;
    @Positive
import org.checkerframework.checker.mustcall.qual.MustCall;
    @Positive
import org.checkerframework.framework.qual.AnnotatedFor;
    @Positive
import java.io.FilePermission;
    @Positive
import java.io.IOException;
    @Positive
import javax.management.DynamicMBean;
    @Positive
import javax.management.MBeanServer;
    @Positive
import javax.management.MBeanServerConnection;
    @Positive
import javax.management.MBeanServerFactory;
    @Positive
import javax.management.MBeanServerPermission;
    @Positive
import javax.management.NotificationEmitter;
    @Positive
import javax.management.ObjectName;
    @Positive
import javax.management.InstanceNotFoundException;
    @Positive
import javax.management.MalformedObjectNameException;
    @Positive
import javax.management.StandardEmitterMBean;
    @Positive
import javax.management.StandardMBean;
    @Positive
import java.security.AccessController;
    @Positive
import java.security.Permission;
    @Positive
import java.security.PrivilegedAction;
    @Positive
import java.security.PrivilegedActionException;
    @Positive
import java.security.PrivilegedExceptionAction;
    @Positive
import java.util.ArrayList;
    @Positive
import java.util.Collection;
    @Positive
import java.util.Collections;
    @Positive
import java.util.HashMap;
    @Positive
import java.util.HashSet;
    @Positive
import java.util.List;
    @Positive
import java.util.Map;
    @Positive
import java.util.Optional;
    @Positive
import java.util.ServiceLoader;
    @Positive
import java.util.Set;
    @Positive
import java.util.stream.Collectors;
    @Positive
import java.util.stream.Stream;
    @Positive
import javax.management.JMX;
    @Positive
import sun.management.Util;
    @Positive
import sun.management.spi.PlatformMBeanProvider;
    @Positive
import sun.management.spi.PlatformMBeanProvider.PlatformComponent;

    @Positive
@AnnotatedFor({ "interning", "mustcall" })
    @Positive
@SuppressWarnings("removal")
    @Positive
@UsesObjectEquals
    @Positive
public class ManagementFactory {

    @Positive
    public static final String CLASS_LOADING_MXBEAN_NAME;

    @Positive
    public static final String COMPILATION_MXBEAN_NAME;

    @Positive
    public static final String MEMORY_MXBEAN_NAME;

    @Positive
    public static final String OPERATING_SYSTEM_MXBEAN_NAME;

    @Positive
    public static final String RUNTIME_MXBEAN_NAME;

    @Positive
    public static final String THREAD_MXBEAN_NAME;

    @Positive
    public static final String GARBAGE_COLLECTOR_MXBEAN_DOMAIN_TYPE;

    @Positive
    public static final String MEMORY_MANAGER_MXBEAN_DOMAIN_TYPE;

    @Positive
    public static final String MEMORY_POOL_MXBEAN_DOMAIN_TYPE;

    @Positive
    public static ClassLoadingMXBean getClassLoadingMXBean();

    @Positive
    public static MemoryMXBean getMemoryMXBean();

    @Positive
    public static ThreadMXBean getThreadMXBean();

    @Positive
    public static RuntimeMXBean getRuntimeMXBean();

    @Positive
    public static CompilationMXBean getCompilationMXBean();

    @Positive
    public static OperatingSystemMXBean getOperatingSystemMXBean();

    @Positive
    public static List<MemoryPoolMXBean> getMemoryPoolMXBeans();

    @Positive
    public static List<MemoryManagerMXBean> getMemoryManagerMXBeans();

    @Positive
    public static List<GarbageCollectorMXBean> getGarbageCollectorMXBeans();

    @Positive
    public static synchronized MBeanServer getPlatformMBeanServer();

    @Positive
    @MustCall({})
    @Positive
    public static <T> T newPlatformMXBeanProxy(MBeanServerConnection connection, String mxbeanName, Class<T> mxbeanInterface) throws java.io.IOException;

    @Positive
    public static <T extends PlatformManagedObject> T getPlatformMXBean(Class<T> mxbeanInterface);

    @Positive
    public static <T extends PlatformManagedObject> List<T> getPlatformMXBeans(Class<T> mxbeanInterface);

    @Positive
    public static <T extends PlatformManagedObject> T getPlatformMXBean(MBeanServerConnection connection, Class<T> mxbeanInterface) throws java.io.IOException;

    @Positive
    public static <T extends PlatformManagedObject> List<T> getPlatformMXBeans(MBeanServerConnection connection, Class<T> mxbeanInterface) throws java.io.IOException;

    @Positive
    public static Set<Class<? extends PlatformManagedObject>> getPlatformManagementInterfaces();

    @Positive
    private static class PlatformMBeanFinder {

    @Positive
        static Map<String, PlatformComponent<?>> getMap();

    @Positive
        static PlatformComponent<?> findFirst(Class<?> mbeanIntf);

    @Positive
        static PlatformComponent<?> findSingleton(Class<?> mbeanIntf);
    @Positive
    }
    @Positive
}

// CFWR semantic augmentation - variant 1
