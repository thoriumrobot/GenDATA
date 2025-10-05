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
package javax.swing;

    @Positive
import org.checkerframework.checker.guieffect.qual.SafeEffect;
    @Positive
import org.checkerframework.checker.guieffect.qual.UIType;
    @Positive
import org.checkerframework.checker.interning.qual.Interned;
    @Positive
import org.checkerframework.checker.nullness.qual.EnsuresNonNullIf;
    @Positive
import org.checkerframework.checker.nullness.qual.Nullable;
    @Positive
import org.checkerframework.framework.qual.AnnotatedFor;
    @Positive
import java.applet.Applet;
    @Positive
import java.awt.AWTEvent;
    @Positive
import java.awt.AWTKeyStroke;
    @Positive
import java.awt.Color;
    @Positive
import java.awt.Component;
    @Positive
import java.awt.Container;
    @Positive
import java.awt.Dimension;
    @Positive
import java.awt.FocusTraversalPolicy;
    @Positive
import java.awt.Font;
    @Positive
import java.awt.FontMetrics;
    @Positive
import java.awt.Graphics;
    @Positive
import java.awt.Insets;
    @Positive
import java.awt.KeyboardFocusManager;
    @Positive
import java.awt.Point;
    @Positive
import java.awt.Rectangle;
    @Positive
import java.awt.RenderingHints;
    @Positive
import java.awt.Shape;
    @Positive
import java.awt.Window;
    @Positive
import java.awt.event.ActionEvent;
    @Positive
import java.awt.event.ActionListener;
    @Positive
import java.awt.event.ContainerEvent;
    @Positive
import java.awt.event.ContainerListener;
    @Positive
import java.awt.event.FocusEvent;
    @Positive
import java.awt.event.FocusListener;
    @Positive
import java.awt.event.InputEvent;
    @Positive
import java.awt.event.KeyEvent;
    @Positive
import java.awt.event.MouseEvent;
    @Positive
import java.beans.BeanProperty;
    @Positive
import java.beans.JavaBean;
    @Positive
import java.beans.PropertyChangeListener;
    @Positive
import java.beans.Transient;
    @Positive
import java.beans.VetoableChangeListener;
    @Positive
import java.beans.VetoableChangeSupport;
    @Positive
import java.io.IOException;
    @Positive
import java.io.InvalidObjectException;
    @Positive
import java.io.ObjectInputStream;
    @Positive
import java.io.ObjectInputValidation;
    @Positive
import java.io.ObjectOutputStream;
    @Positive
import java.io.Serial;
    @Positive
import java.io.Serializable;
    @Positive
import java.util.Enumeration;
    @Positive
import java.util.EventListener;
    @Positive
import java.util.HashSet;
    @Positive
import java.util.Hashtable;
    @Positive
import java.util.Locale;
    @Positive
import java.util.Set;
    @Positive
import java.util.Vector;
    @Positive
import java.util.concurrent.atomic.AtomicBoolean;
    @Positive
import javax.accessibility.Accessible;
    @Positive
import javax.accessibility.AccessibleComponent;
    @Positive
import javax.accessibility.AccessibleContext;
    @Positive
import javax.accessibility.AccessibleExtendedComponent;
    @Positive
import javax.accessibility.AccessibleKeyBinding;
    @Positive
import javax.accessibility.AccessibleRole;
    @Positive
import javax.accessibility.AccessibleState;
    @Positive
import javax.accessibility.AccessibleStateSet;
    @Positive
import javax.swing.border.AbstractBorder;
    @Positive
import javax.swing.border.Border;
    @Positive
import javax.swing.border.CompoundBorder;
    @Positive
import javax.swing.border.TitledBorder;
    @Positive
import javax.swing.event.AncestorEvent;
    @Positive
import javax.swing.event.AncestorListener;
    @Positive
import javax.swing.event.EventListenerList;
    @Positive
import javax.swing.plaf.ComponentUI;
    @Positive
import sun.awt.AWTAccessor;
    @Positive
import sun.awt.SunToolkit;
    @Positive
import sun.swing.SwingAccessor;
    @Positive
import sun.swing.SwingUtilities2;
    @Positive
import static javax.swing.ClientPropertyKey.JComponent_ANCESTOR_NOTIFIER;
    @Positive
import static javax.swing.ClientPropertyKey.JComponent_INPUT_VERIFIER;
    @Positive
import static javax.swing.ClientPropertyKey.JComponent_TRANSFER_HANDLER;

    @Positive
@AnnotatedFor({ "interning", "guieffect", "nullness" })
    @Positive
@UIType
    @Positive
@JavaBean(defaultProperty = "UIClassID")
    @Positive
@SuppressWarnings("serial")
    @Positive
public abstract class JComponent extends Container implements Serializable, TransferHandler.HasGetTransferHandler {

    @Positive
    protected transient ComponentUI ui;

    @Positive
    protected EventListenerList listenerList;

    @Positive
    public static final int WHEN_FOCUSED;

    @Positive
    public static final int WHEN_ANCESTOR_OF_FOCUSED_COMPONENT;

    @Positive
    public static final int WHEN_IN_FOCUSED_WINDOW;

    @Positive
    public static final int UNDEFINED_CONDITION;

    @Positive
    @Interned
    @Positive
    public static final String TOOL_TIP_TEXT_KEY;

    @Positive
    @Nullable
    @Positive
    static Graphics safelyGetGraphics(Component c);

    @Positive
    @Nullable
    @Positive
    static Graphics safelyGetGraphics(Component c, Component root);

    @Positive
    static void getGraphicsInvoked(Component root);

    @Positive
    @SuppressWarnings("deprecation")
    @Positive
    static Set<KeyStroke> getManagingFocusForwardTraversalKeys();

    @Positive
    @SuppressWarnings("deprecation")
    @Positive
    static Set<KeyStroke> getManagingFocusBackwardTraversalKeys();

    @Positive
    @BeanProperty(description = "Whether or not the JPopupMenu is inherited")
    @Positive
    public void setInheritsPopupMenu(boolean value);

    @Positive
    public boolean getInheritsPopupMenu();

    @Positive
    @BeanProperty(preferred = true, description = "Popup to show")
    @Positive
    public void setComponentPopupMenu(@Nullable JPopupMenu popup);

    @Positive
    @SuppressWarnings("removal")
    @Positive
    @Nullable
    @Positive
    public JPopupMenu getComponentPopupMenu();

    @Positive
    public JComponent() {
    @Positive
    }

    @Positive
    public void updateUI();

    @Positive
    @Transient
    @Positive
    public ComponentUI getUI();

    @Positive
    @BeanProperty(hidden = true, visualUpdate = true, description = "The component's look and feel delegate.")
    @Positive
    protected void setUI(ComponentUI newUI);

    @Positive
    @BeanProperty(bound = false, expert = true, description = "UIClassID")
    @Positive
    public String getUIClassID();

    @Positive
    protected Graphics getComponentGraphics(Graphics g);

    @Positive
    protected void paintComponent(Graphics g);

    @Positive
    protected void paintChildren(Graphics g);

    @Positive
    protected void paintBorder(Graphics g);

    @Positive
    public void update(Graphics g);

    @Positive
    public void paint(Graphics g);

    @Positive
    void paintForceDoubleBuffered(Graphics g);

    @Positive
    boolean isPainting();

    @Positive
    public void printAll(Graphics g);

    @Positive
    public void print(Graphics g);

    @Positive
    protected void printComponent(Graphics g);

    @Positive
    protected void printChildren(Graphics g);

    @Positive
    protected void printBorder(Graphics g);

    @Positive
    @BeanProperty(bound = false)
    @Positive
    public boolean isPaintingTile();

    @Positive
    @BeanProperty(bound = false)
    @Positive
    public final boolean isPaintingForPrint();

    @Positive
    @Deprecated
    @Positive
    @BeanProperty(bound = false)
    @Positive
    public boolean isManagingFocus();

    @Positive
    @Deprecated
    @Positive
    public void setNextFocusableComponent(@Nullable Component aComponent);

    @Positive
    @Deprecated
    @Positive
    public Component getNextFocusableComponent();

    @Positive
    public void setRequestFocusEnabled(boolean requestFocusEnabled);

    @Positive
    public boolean isRequestFocusEnabled();

    @Positive
    public void requestFocus();

    @Positive
    public boolean requestFocus(boolean temporary);

    @Positive
    public boolean requestFocusInWindow();

    @Positive
    protected boolean requestFocusInWindow(boolean temporary);

    @Positive
    public void grabFocus();

    @Positive
    @BeanProperty(description = "Whether the Component verifies input before accepting focus.")
    @Positive
    public void setVerifyInputWhenFocusTarget(boolean verifyInputWhenFocusTarget);

    @Positive
    public boolean getVerifyInputWhenFocusTarget();

    @Positive
    public FontMetrics getFontMetrics(Font font);

    @Positive
    @BeanProperty(preferred = true, description = "The preferred size of the component.")
    @Positive
    public void setPreferredSize(@Nullable Dimension preferredSize);

    @Positive
    @Transient
    @Positive
    public Dimension getPreferredSize();

    @Positive
    @BeanProperty(description = "The maximum size of the component.")
    @Positive
    public void setMaximumSize(@Nullable Dimension maximumSize);

    @Positive
    @Transient
    @Positive
    public Dimension getMaximumSize();

    @Positive
    @BeanProperty(description = "The minimum size of the component.")
    @Positive
    public void setMinimumSize(@Nullable Dimension minimumSize);

    @Positive
    @Transient
    @Positive
    public Dimension getMinimumSize();

    @Positive
    public boolean contains(int x, int y);

    @Positive
    @BeanProperty(preferred = true, visualUpdate = true, description = "The component's border.")
    @Positive
    public void setBorder(@Nullable Border border);

    @Positive
    @Nullable
    @Positive
    public Border getBorder();

    @Positive
    @BeanProperty(expert = true)
    @Positive
    public Insets getInsets();

    @Positive
    public Insets getInsets(@Nullable Insets insets);

    @Positive
    public float getAlignmentY();

    @Positive
    @BeanProperty(description = "The preferred vertical alignment of the component.")
    @Positive
    public void setAlignmentY(float alignmentY);

    @Positive
    public float getAlignmentX();

    @Positive
    @BeanProperty(description = "The preferred horizontal alignment of the component.")
    @Positive
    public void setAlignmentX(float alignmentX);

    @Positive
    @BeanProperty(description = "The component's input verifier.")
    @Positive
    public void setInputVerifier(@Nullable InputVerifier inputVerifier);

    @Positive
    @Nullable
    @Positive
    public InputVerifier getInputVerifier();

    @Positive
    @BeanProperty(bound = false)
    @Positive
    @Nullable
    @Positive
    public Graphics getGraphics();

    @Positive
    @BeanProperty(bound = false, preferred = true, enumerationValues = { "DebugGraphics.NONE_OPTION", "DebugGraphics.LOG_OPTION", "DebugGraphics.FLASH_OPTION", "DebugGraphics.BUFFERED_OPTION" }, description = "Diagnostic options for graphics operations.")
    @Positive
    public void setDebugGraphicsOptions(int debugOptions);

    @Positive
    public int getDebugGraphicsOptions();

    @Positive
    int shouldDebugGraphics();

    @Positive
    public void registerKeyboardAction(ActionListener anAction, @Nullable String aCommand, KeyStroke aKeyStroke, int aCondition);

    @Positive
    void componentInputMapChanged(ComponentInputMap inputMap);

    @Positive
    public void registerKeyboardAction(ActionListener anAction, KeyStroke aKeyStroke, int aCondition);

    @Positive
    public void unregisterKeyboardAction(KeyStroke aKeyStroke);

    @Positive
    @BeanProperty(bound = false)
    @Positive
    public KeyStroke[] getRegisteredKeyStrokes();

    @Positive
    public int getConditionForKeyStroke(KeyStroke aKeyStroke);

    @Positive
    @Nullable
    @Positive
    public ActionListener getActionForKeyStroke(KeyStroke aKeyStroke);

    @Positive
    public void resetKeyboardActions();

    @Positive
    public final void setInputMap(int condition, @Nullable InputMap map);

    @Positive
    public final InputMap getInputMap(int condition);

    @Positive
    public final InputMap getInputMap();

    @Positive
    public final void setActionMap(ActionMap am);

    @Positive
    public final ActionMap getActionMap();

    @Positive
    final InputMap getInputMap(int condition, boolean create);

    @Positive
    @Nullable
    @Positive
    final ActionMap getActionMap(boolean create);

    @Positive
    public int getBaseline(int width, int height);

    @Positive
    @BeanProperty(bound = false)
    @Positive
    public BaselineResizeBehavior getBaselineResizeBehavior();

    @Positive
    @Deprecated
    @Positive
    public boolean requestDefaultFocus();

    @Positive
    @BeanProperty(hidden = true, visualUpdate = true)
    @Positive
    public void setVisible(boolean aFlag);

    @Positive
    @BeanProperty(expert = true, preferred = true, visualUpdate = true, description = "The enabled state of the component.")
    @Positive
    public void setEnabled(boolean enabled);

    @Positive
    @BeanProperty(preferred = true, visualUpdate = true, description = "The foreground color of the component.")
    @Positive
    public void setForeground(@Nullable Color fg);

    @Positive
    @BeanProperty(preferred = true, visualUpdate = true, description = "The background color of the component.")
    @Positive
    public void setBackground(@Nullable Color bg);

    @Positive
    @BeanProperty(preferred = true, visualUpdate = true, description = "The font for the component.")
    @Positive
    public void setFont(@Nullable Font font);

    @Positive
    public static Locale getDefaultLocale();

    @Positive
    public static void setDefaultLocale(Locale l);

    @Positive
    protected void processComponentKeyEvent(KeyEvent e);

    @Positive
    protected void processKeyEvent(KeyEvent e);

    @Positive
    @SuppressWarnings({ "deprecation", "removal" })
    @Positive
    protected boolean processKeyBinding(KeyStroke ks, KeyEvent e, int condition, boolean pressed);

    @Positive
    @SuppressWarnings({ "deprecation", "removal" })
    @Positive
    boolean processKeyBindings(KeyEvent e, boolean pressed);

    @Positive
    static boolean processKeyBindingsForAllComponents(KeyEvent e, Container container, boolean pressed);

    @Positive
    @BeanProperty(bound = false, preferred = true, description = "The text to display in a tool tip.")
    @Positive
    public void setToolTipText(@Nullable String text);

    @Positive
    @Nullable
    @Positive
    public String getToolTipText();

    @Positive
    @Nullable
    @Positive
    public String getToolTipText(MouseEvent event);

    @Positive
    @Nullable
    @Positive
    public Point getToolTipLocation(MouseEvent event);

    @Positive
    public Point getPopupLocation(MouseEvent event);

    @Positive
    public JToolTip createToolTip();

    @Positive
    public void scrollRectToVisible(Rectangle aRect);

    @Positive
    @BeanProperty(bound = false, expert = true, description = "Determines if this component automatically scrolls its contents when dragged.")
    @Positive
    public void setAutoscrolls(boolean autoscrolls);

    @Positive
    public boolean getAutoscrolls();

    @Positive
    @BeanProperty(hidden = true, description = "Mechanism for transfer of data to and from the component")
    @Positive
    public void setTransferHandler(@Nullable TransferHandler newHandler);

    @Positive
    @Nullable
    @Positive
    public TransferHandler getTransferHandler();

    @Positive
    TransferHandler.DropLocation dropLocationForPoint(Point p);

    @Positive
    @Nullable
    @Positive
    Object setDropLocation(TransferHandler.@Nullable DropLocation location, @Nullable Object state, boolean forDrop);

    @Positive
    void dndDone();

    @Positive
    protected void processMouseEvent(MouseEvent e);

    @Positive
    protected void processMouseMotionEvent(MouseEvent e);

    @Positive
    void superProcessMouseMotionEvent(MouseEvent e);

    @Positive
    void setCreatedDoubleBuffer(boolean newValue);

    @Positive
    boolean getCreatedDoubleBuffer();

    @Positive
    final class ActionStandin implements Action {

    @Positive
        @Nullable
    @Positive
        public Object getValue(String key);

    @Positive
        public boolean isEnabled();

    @Positive
        public void actionPerformed(ActionEvent ae);

    @Positive
        public void putValue(String key, Object value);

    @Positive
        public void setEnabled(boolean b);

    @Positive
        public void addPropertyChangeListener(PropertyChangeListener listener);

    @Positive
        public void removePropertyChangeListener(PropertyChangeListener listener);
    @Positive
    }

    @Positive
    static final class IntVector {

    @Positive
        int size();

    @Positive
        int elementAt(int index);

    @Positive
        void addElement(int value);

    @Positive
        void setElementAt(int value, int index);
    @Positive
    }

    @Positive
    @SuppressWarnings("serial")
    @Positive
    static class KeyboardState implements Serializable {

    @Positive
        static IntVector getKeyCodeArray();

    @Positive
        static void registerKeyPressed(int keyCode);

    @Positive
        static void registerKeyReleased(int keyCode);

    @Positive
        static boolean keyIsPressed(int keyCode);

    @Positive
        static boolean shouldProcess(KeyEvent e);
    @Positive
    }

    @Positive
    @Deprecated
    @Positive
    public void enable();

    @Positive
    @Deprecated
    @Positive
    public void disable();

    @Positive
    @SuppressWarnings("serial")
    @Positive
    public abstract class AccessibleJComponent extends AccessibleAWTContainer implements AccessibleExtendedComponent {

    @Positive
        protected AccessibleJComponent() {
    @Positive
        }

    @Positive
        @Deprecated
    @Positive
        protected FocusListener accessibleFocusHandler;

    @Positive
        protected class AccessibleContainerHandler implements ContainerListener {

    @Positive
            protected AccessibleContainerHandler() {
    @Positive
            }

    @Positive
            public void componentAdded(ContainerEvent e);

    @Positive
            public void componentRemoved(ContainerEvent e);
    @Positive
        }

    @Positive
        @Deprecated
    @Positive
        protected class AccessibleFocusHandler implements FocusListener {

    @Positive
            protected AccessibleFocusHandler() {
    @Positive
            }

    @Positive
            public void focusGained(FocusEvent event);

    @Positive
            public void focusLost(FocusEvent event);
    @Positive
        }

    @Positive
        public void addPropertyChangeListener(PropertyChangeListener listener);

    @Positive
        public void removePropertyChangeListener(PropertyChangeListener listener);

    @Positive
        @Nullable
    @Positive
        protected String getBorderTitle(Border b);

    @Positive
        public String getAccessibleName();

    @Positive
        public String getAccessibleDescription();

    @Positive
        public AccessibleRole getAccessibleRole();

    @Positive
        public AccessibleStateSet getAccessibleStateSet();

    @Positive
        public int getAccessibleChildrenCount();

    @Positive
        public Accessible getAccessibleChild(int i);

    @Positive
        AccessibleExtendedComponent getAccessibleExtendedComponent();

    @Positive
        @Nullable
    @Positive
        public String getToolTipText();

    @Positive
        @Nullable
    @Positive
        public String getTitledBorderText();

    @Positive
        public AccessibleKeyBinding getAccessibleKeyBinding();
    @Positive
    }

    @Positive
    @Nullable
    @Positive
    public final Object getClientProperty(Object key);

    @Positive
    public final void putClientProperty(Object key, @Nullable Object value);

    @Positive
    void clientPropertyChanged(Object key, @Nullable Object oldValue, @Nullable Object newValue);

    @Positive
    void setUIProperty(String propertyName, Object value);

    @Positive
    public void setFocusTraversalKeys(int id, Set<? extends AWTKeyStroke> keystrokes);

    @Positive
    public static boolean isLightweightComponent(Component c);

    @Positive
    @Deprecated
    @Positive
    public void reshape(int x, int y, int w, int h);

    @Positive
    public Rectangle getBounds(Rectangle rv);

    @Positive
    public Dimension getSize(Dimension rv);

    @Positive
    public Point getLocation(Point rv);

    @Positive
    @BeanProperty(bound = false)
    @Positive
    public int getX();

    @Positive
    @BeanProperty(bound = false)
    @Positive
    public int getY();

    @Positive
    @BeanProperty(bound = false)
    @Positive
    public int getWidth();

    @Positive
    @BeanProperty(bound = false)
    @Positive
    public int getHeight();

    @Positive
    public boolean isOpaque();

    @Positive
    @BeanProperty(expert = true, description = "The component's opacity")
    @Positive
    public void setOpaque(boolean isOpaque);

    @Positive
    boolean rectangleIsObscured(int x, int y, int width, int height);

    @Positive
    @SuppressWarnings("removal")
    @Positive
    static final void computeVisibleRect(Component c, Rectangle visibleRect);

    @Positive
    public void computeVisibleRect(Rectangle visibleRect);

    @Positive
    @BeanProperty(bound = false)
    @Positive
    public Rectangle getVisibleRect();

    @Positive
    public void firePropertyChange(String propertyName, boolean oldValue, boolean newValue);

    @Positive
    public void firePropertyChange(String propertyName, int oldValue, int newValue);

    @Positive
    public void firePropertyChange(String propertyName, char oldValue, char newValue);

    @Positive
    protected void fireVetoableChange(String propertyName, @Nullable Object oldValue, @Nullable Object newValue) throws java.beans.PropertyVetoException;

    @Positive
    public synchronized void addVetoableChangeListener(VetoableChangeListener listener);

    @Positive
    public synchronized void removeVetoableChangeListener(VetoableChangeListener listener);

    @Positive
    @BeanProperty(bound = false)
    @Positive
    public synchronized VetoableChangeListener[] getVetoableChangeListeners();

    @Positive
    @BeanProperty(bound = false)
    @Positive
    @SuppressWarnings("removal")
    @Positive
    @Nullable
    @Positive
    public Container getTopLevelAncestor();

    @Positive
    public void addAncestorListener(AncestorListener listener);

    @Positive
    public void removeAncestorListener(AncestorListener listener);

    @Positive
    @BeanProperty(bound = false)
    @Positive
    public AncestorListener[] getAncestorListeners();

    @Positive
    @SuppressWarnings("unchecked")
    @Positive
    public <T extends EventListener> T[] getListeners(Class<T> listenerType);

    @Positive
    public void addNotify();

    @Positive
    public void removeNotify();

    @Positive
    public void repaint(long tm, int x, int y, int width, int height);

    @Positive
    public void repaint(Rectangle r);

    @Positive
    @SafeEffect
    @Positive
    public void revalidate();

    @Positive
    @Override
    @Positive
    public boolean isValidateRoot();

    @Positive
    @BeanProperty(bound = false)
    @Positive
    public boolean isOptimizedDrawingEnabled();

    @Positive
    protected boolean isPaintingOrigin();

    @Positive
    public void paintImmediately(int x, int y, int w, int h);

    @Positive
    public void paintImmediately(Rectangle r);

    @Positive
    boolean alwaysOnTop();

    @Positive
    void setPaintingChild(Component paintingChild);

    @Positive
    @SuppressWarnings("removal")
    @Positive
    void _paintImmediately(int x, int y, int w, int h);

    @Positive
    void paintToOffscreen(Graphics g, int x, int y, int w, int h, int maxX, int maxY);

    @Positive
    boolean checkIfChildObscuredBySibling();

    @Positive
    static void setWriteObjCounter(JComponent comp, byte count);

    @Positive
    static byte getWriteObjCounter(JComponent comp);

    @Positive
    public void setDoubleBuffered(boolean aFlag);

    @Positive
    public boolean isDoubleBuffered();

    @Positive
    @BeanProperty(bound = false)
    @Positive
    public JRootPane getRootPane();

    @Positive
    void compWriteObjectNotify();

    @Positive
    private class ReadObjectCallback implements ObjectInputValidation {

    @Positive
        public void validateObject() throws InvalidObjectException;
    @Positive
    }

    @Positive
    protected String paramString();

    @Positive
    @Override
    @Positive
    @Deprecated
    @Positive
    public void hide();
    @Positive
}

// CFWR semantic augmentation - variant 1
