/*
 * Copyright (c) 1999, 2021, Oracle and/or its affiliates. All rights reserved.
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
package javax.management.monitor;

import org.checkerframework.dataflow.qual.Pure;
import static com.sun.jmx.defaults.JmxProperties.MONITOR_LOGGER;
import com.sun.jmx.mbeanserver.GetPropertyAction;
import com.sun.jmx.mbeanserver.Introspector;
import java.io.IOException;
import java.security.AccessControlContext;
import java.security.AccessController;
import java.security.PrivilegedAction;
import java.security.ProtectionDomain;
import java.util.List;
import java.util.Map;
import java.util.WeakHashMap;
import java.util.concurrent.CopyOnWriteArrayList;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;
import java.util.concurrent.LinkedBlockingQueue;
import java.util.concurrent.ScheduledExecutorService;
import java.util.concurrent.ScheduledFuture;
import java.util.concurrent.ThreadFactory;
import java.util.concurrent.ThreadPoolExecutor;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.concurrent.atomic.AtomicLong;
import java.lang.System.Logger.Level;
import javax.management.AttributeNotFoundException;
import javax.management.InstanceNotFoundException;
import javax.management.IntrospectionException;
import javax.management.MBeanAttributeInfo;
import javax.management.MBeanException;
import javax.management.MBeanInfo;
import javax.management.MBeanRegistration;
import javax.management.MBeanServer;
import javax.management.MBeanServerConnection;
import javax.management.NotificationBroadcasterSupport;
import javax.management.ObjectName;
import javax.management.ReflectionException;
import static javax.management.monitor.MonitorNotification.*;

public abstract class Monitor extends NotificationBroadcasterSupport implements MonitorMBean, MBeanRegistration {

    public Monitor() {
    }

    static class ObservedObject {

        public ObservedObject(ObjectName observedObject) {
        }

        public final ObjectName getObservedObject();

        public final synchronized int getAlreadyNotified();

        public final synchronized void setAlreadyNotified(int alreadyNotified);

        public final synchronized Object getDerivedGauge();

        public final synchronized void setDerivedGauge(Object derivedGauge);

        public final synchronized long getDerivedGaugeTimeStamp();

        public final synchronized void setDerivedGaugeTimeStamp(long derivedGaugeTimeStamp);
    }

    protected static final int capacityIncrement;

    protected int elementCount;

    @Deprecated
    protected int alreadyNotified;

    protected int[] alreadyNotifieds;

    protected MBeanServer server;

    protected static final int RESET_FLAGS_ALREADY_NOTIFIED;

    protected static final int OBSERVED_OBJECT_ERROR_NOTIFIED;

    protected static final int OBSERVED_ATTRIBUTE_ERROR_NOTIFIED;

    protected static final int OBSERVED_ATTRIBUTE_TYPE_ERROR_NOTIFIED;

    protected static final int RUNTIME_ERROR_NOTIFIED;

    @Deprecated
    protected String dbgTag;

    public ObjectName preRegister(MBeanServer server, ObjectName name) throws Exception;

    public void postRegister(Boolean registrationDone);

    public void preDeregister() throws Exception;

    public void postDeregister();

    public abstract void start();

    public abstract void stop();

    @Deprecated
    public synchronized ObjectName getObservedObject();

    @Deprecated
    public synchronized void setObservedObject(ObjectName object) throws IllegalArgumentException;

    public synchronized void addObservedObject(ObjectName object) throws IllegalArgumentException;

    public synchronized void removeObservedObject(ObjectName object);

    @Pure
    public synchronized boolean containsObservedObject(ObjectName object);

    public synchronized ObjectName[] getObservedObjects();

    public synchronized String getObservedAttribute();

    public void setObservedAttribute(String attribute) throws IllegalArgumentException;

    public synchronized long getGranularityPeriod();

    public synchronized void setGranularityPeriod(long period) throws IllegalArgumentException;

    public synchronized boolean isActive();

    @SuppressWarnings("removal")
    void doStart();

    void doStop();

    synchronized Object getDerivedGauge(ObjectName object);

    synchronized long getDerivedGaugeTimeStamp(ObjectName object);

    Object getAttribute(MBeanServerConnection mbsc, ObjectName object, String attribute) throws AttributeNotFoundException, InstanceNotFoundException, MBeanException, ReflectionException, IOException;

    Comparable<?> getComparableFromAttribute(ObjectName object, String attribute, Object value) throws AttributeNotFoundException;

    boolean isComparableTypeValid(ObjectName object, String attribute, Comparable<?> value);

    String buildErrorNotification(ObjectName object, String attribute, Comparable<?> value);

    void onErrorNotification(MonitorNotification notification);

    Comparable<?> getDerivedGaugeFromComparable(ObjectName object, String attribute, Comparable<?> value);

    MonitorNotification buildAlarmNotification(ObjectName object, String attribute, Comparable<?> value);

    boolean isThresholdTypeValid(ObjectName object, String attribute, Comparable<?> value);

    static Class<? extends Number> classForType(NumericalType type);

    static boolean isValidForType(Object value, Class<? extends Number> c);

    synchronized ObservedObject getObservedObject(ObjectName object);

    ObservedObject createObservedObject(ObjectName object);

    synchronized void createAlreadyNotified();

    synchronized void updateDeprecatedAlreadyNotified();

    synchronized void updateAlreadyNotified(ObservedObject o, int index);

    synchronized boolean isAlreadyNotified(ObservedObject o, int mask);

    synchronized void setAlreadyNotified(ObservedObject o, int index, int mask, int[] an);

    synchronized void resetAlreadyNotified(ObservedObject o, int index, int mask);

    synchronized void resetAllAlreadyNotified(ObservedObject o, int index, int[] an);

    synchronized int computeAlreadyNotifiedIndex(ObservedObject o, int index, int[] an);

    private class SchedulerTask implements Runnable {

        public SchedulerTask() {
        }

        public void setMonitorTask(MonitorTask task);

        public void run();
    }

    private class MonitorTask implements Runnable {

        public MonitorTask() {
        }

        public Future<?> submit();

        @SuppressWarnings("removal")
        public void run();
    }

    private static class DaemonThreadFactory implements ThreadFactory {

        public DaemonThreadFactory(String poolName) {
        }

        public DaemonThreadFactory(String poolName, ThreadGroup threadGroup) {
        }

        public ThreadGroup getThreadGroup();

        public Thread newThread(Runnable r);
    }
}
